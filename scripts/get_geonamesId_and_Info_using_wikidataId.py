from rdflib import Graph, Namespace, URIRef
from rdflib.plugins.sparql import prepareQuery
from xml.dom import minidom
from tqdm import tqdm
import os
import json
import os.path
from SPARQLWrapper import SPARQLWrapper, JSON

# set up your own proxy
proxy = "http://:port"
os.environ['http_proxy'] = proxy
os.environ['HTTP_PROXY'] = proxy
os.environ['https_proxy'] = proxy
os.environ['HTTPS_PROXY'] = proxy

def get_annotated_dict_infos_recit_text_from_json_file(file_path):
    with open(file_path, 'r') as file:
        dict_annotated_infos_mention_entity = json.load(file)
    return dict_annotated_infos_mention_entity


def get_geonames_id(string_wikidataId):
    # url to make sparql queries on wikidata
    url = "https://query.wikidata.org/sparql"
    sparql = SPARQLWrapper(url)
    string_query = """
    SELECT ?id_geonames
    WHERE
    {
        wd:%s wdt:P1566 ?id_geonames
    }
    """
    query = string_query % string_wikidataId
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    try:
        result = sparql.query().convert()
        return result['results']['bindings'][0]['id_geonames']['value']
    except IndexError:
        return "None"


def get_geonamesId_from_list_entity_with_wikidataId(list_dict_entity_with_wikidataId):
    list_dict_entity_with_wikidataId_and_geonamesId = []

    for dict_entity_with_wikidataId in tqdm(list_dict_entity_with_wikidataId):
        dict_entity_with_wikidataId_and_geonamesId = {"entity_group": dict_entity_with_wikidataId["entity_group"],
                                                      "ner_score": dict_entity_with_wikidataId["ner_score"],
                                                      "word": dict_entity_with_wikidataId["word"],
                                                      "wikidataId": dict_entity_with_wikidataId["wikidataId"],
                                                      "geonamesId": get_geonames_id(
                                                          dict_entity_with_wikidataId["wikidataId"]),
                                                      "start": dict_entity_with_wikidataId["start"],
                                                      "end": dict_entity_with_wikidataId["end"]}

        list_dict_entity_with_wikidataId_and_geonamesId.append(dict_entity_with_wikidataId_and_geonamesId)

    return list_dict_entity_with_wikidataId_and_geonamesId


def query_geonames_n_triples(rdf_graph, namespaces, str_geonames_id):
    # define namespaces used in the RDF data
    gn = Namespace("http://www.geonames.org/ontology#")
    wgs84_pos = Namespace("http://www.w3.org/2003/01/geo/wgs84_pos#")

    # GeoNames ID to retrieve longitude and latitude for
    geonames_url = "https://sws.geonames.org/" + str_geonames_id + "/"

    # define the SPARQL query to retrieve longitude, latitude, GeoNames name, official name, and alternate names based on GeoNames ID
    query = prepareQuery(
        """
        SELECT ?latitude ?longitude ?gnName ?officialName ?alternateName
        WHERE {
            ?feature rdf:type gn:Feature .
            ?feature wgs84_pos:lat ?latitude .
            ?feature wgs84_pos:long ?longitude .
            OPTIONAL { ?feature gn:name ?gnName }
            OPTIONAL { ?feature gn:officialName ?officialName }
            OPTIONAL { ?feature gn:alternateName ?alternateName }
            FILTER(?feature = ?id)
        }
        """,
        initNs={"rdf": namespaces['rdf'], "gn": gn, "wgs84_pos": wgs84_pos},
    )

    # execute the SPARQL query and store results
    print("Executing SPARQL query...")
    results = rdf_graph.query(query, initBindings={"id": URIRef(geonames_url)})
    dic_location_info = {}
    for row in results:
        location_name = row.gnName.value if row.gnName else "None"
        location_official_name = row.officialName.value if row.officialName else "None"
        location_alternate_name = row.alternateName.value if row.alternateName else "None"
        location_latitude = float(row.latitude.value) if row.latitude.value else "None"
        location_longitude = float(row.longitude.value) if row.longitude.value else "None"
        dic_location_info = {"GeoName": location_name,
                             "Official Name": location_official_name,
                             "Alternate Names": location_alternate_name,
                             "Latitude": location_latitude,
                             "Longitude": location_longitude}

    return dic_location_info


# function to estimate file size
def get_file_size(filename):
    return os.path.getsize(filename)


def load_rdf_with_progress(filename):
    g = Graph()
    # extract namespaces from the RDFLib graph
    namespaces = dict(g.namespace_manager.namespaces())
    file_size = get_file_size(filename)
    bytes_read = 0

    with open(filename, "rb") as f:
        xml_content = f.read()
        xml_dom = minidom.parseString(xml_content)
        rdf_data = xml_dom.toxml()

        g.parse(data=rdf_data, format="xml")
        bytes_read += len(rdf_data)
        progress = (bytes_read / file_size) * 100
        print(f"Loading progress: {progress:.2f}%")

    return g, namespaces


def get_geonamesInfo_from_list_entity_with_wikidataId_and_geonamesId(list_mention_entity_with_wikidataId_and_geonamesId, graph_rdf, name_spaces):
    list_mention_entity_with_wikidataId_and_geonamesId_info = []
    for dict_entity_with_wikidataId_info in list_mention_entity_with_wikidataId_and_geonamesId:

        if dict_entity_with_wikidataId_info["geonamesId"] != "None":
            response_geonames_info = query_geonames_n_triples(rdf_graph=graph_rdf,
                                                              str_geonames_id=dict_entity_with_wikidataId_info["geonamesId"],
                                                              namespaces=name_spaces)
            dict_entity_with_wikidataId_and_geonamesId = {
                "entity_group": dict_entity_with_wikidataId_info["entity_group"],
                "ner_score": dict_entity_with_wikidataId_info["ner_score"],
                "word": dict_entity_with_wikidataId_info["word"],
                "wikidataId": dict_entity_with_wikidataId_info["wikidataId"],
                "geonamesId": dict_entity_with_wikidataId_info["geonamesId"],
                "latitude": response_geonames_info.get("Latitude", "None"),
                "longitude": response_geonames_info.get("Longitude", "None"),
                "start": dict_entity_with_wikidataId_info["start"],
                "end": dict_entity_with_wikidataId_info["end"]}
            list_mention_entity_with_wikidataId_and_geonamesId_info.append(dict_entity_with_wikidataId_and_geonamesId)
        else:
            dict_entity_with_wikidataId_and_geonamesId = {
                "entity_group": dict_entity_with_wikidataId_info["entity_group"],
                "ner_score": dict_entity_with_wikidataId_info["ner_score"],
                "word": dict_entity_with_wikidataId_info["word"],
                "wikidataId": dict_entity_with_wikidataId_info["wikidataId"],
                "geonamesId": dict_entity_with_wikidataId_info["geonamesId"],
                "latitude": "None",
                "longitude": "None",
                "start": dict_entity_with_wikidataId_info["start"],
                "end": dict_entity_with_wikidataId_info["end"]}
            list_mention_entity_with_wikidataId_and_geonamesId_info.append(dict_entity_with_wikidataId_and_geonamesId)

    return list_mention_entity_with_wikidataId_and_geonamesId_info
