import xml.dom.minidom
import re

def clean_dump_line(line):
    line = re.sub(">( *)<", r">\n\1<", line)
    # remove the alternateName, officialName and shortName beside fr and en
    line = re.sub(
        "<gn:(alternateName|officialName|shortName) xml:lang=\"(?!fr|en).*\">.*</gn:(alternateName|officialName|shortName)>",
        "", line)
    # remove empty lines
    line = re.sub("\n\s*\n", "\n", line, flags=re.MULTILINE)
    line = re.sub("</rdf:RDF>", "", line)

    return line


with open("../dataset/GeoNames/all-geonames-rdf.txt", "r", encoding='utf-8') as r:
    with open("../Geonames/clean_all_geonames_rdf_african_countries.txt", "w", True, encoding='utf-8') as w:
        # ignore first line
        r.readline()
        # second line
        l = r.readline()
        l = clean_dump_line(l)
        w.write(l)

        # passe = True
        for line in r:
            if line.startswith("https://sws.geonames.org/"):
                continue
            print(line)
            doc = xml.dom.minidom.parseString(line)
            cc = doc.getElementsByTagName('gn:countryCode')
            if len(cc) == 0:
                continue
            cc = cc[0].firstChild.data
            # list of related country codes for migration visualization
            """
            if cc not in (
                    "AF", "AL", "DZ", "BE", "BJ", "BA", "BG", "BF", "CM", "CF", "TD", "CD", "CG", "CI", "HR", "GQ",
                    "ET", "FR", "GM", "DE", "GH", "GR", "GN", "GW", "HU", "IR", "IQ", "IT", "LY", "ML", "MR", "ME",
                    "MA", "NE", "NG", "MK", "PK", "PL", "RO", "RU", "SA", "SN", "RS", "SK", "SI", "SO", "ZA", "SS",
                    "ES", "SD", "CH", "SY", "TG", "TN", "TR", "UA", "UG", "ZM"):
            """
            if cc not in ("DZ", "MA", "SN", "GN", "GW", "CI", "BF", "NG", "MR"):
                # print("ignoring country", cc, file=sys.stderr)
                continue
            line = re.sub(
                "\<\?xml version=\"1\.0\" encoding=\"UTF-8\" standalone=\"no\"\?\>\<rdf\:RDF xmlns\:cc=\"http\://creativecommons\.org/ns#\" xmlns\:dcterms=\"http\://purl\.org/dc/terms/\" xmlns\:foaf=\"http\://xmlns\.com/foaf/0\.1/\" xmlns\:gn=\"http\://www\.geonames\.org/ontology#\" xmlns\:owl=\"http\://www\.w3\.org/2002/07/owl#\" xmlns\:rdf=\"http\://www\.w3\.org/1999/02/22-rdf-syntax-ns#\" xmlns\:rdfs=\"http\://www\.w3\.org/2000/01/rdf-schema#\" xmlns\:wgs84_pos=\"http\://www\.w3\.org/2003/01/geo/wgs84_pos#\"\>",
                "", line)
            line = clean_dump_line(line)
            w.write(line)
        w.write("</rdf:RDF>")

# f = open("../dataset/all-geonames-rdf.txt")
# i = 0

'''
while True:
    # if (i+1) % 10_000 == 0:
    #     break
    line = f.readline()  # ignore path
    if line == "":
        break
    line = f.readline()
    if line == "":
        break
    print(line)
    doc = xml.dom.minidom.parseString(line)
    cc = doc.getElementsByTagName('gn:countryCode')
    if len(cc) == 0:
        continue
    cc = cc[0].firstChild.data
    if cc not in ("AF", "AL", "DZ", "BE", "BJ", "BA", "BG", "BF", "CM", "CF", "TD", "CD", "CG", "CI", "HR", "GQ",
                  "ET", "FR", "GM", "DE", "GH", "GR", "GN", "GW", "HU", "IR", "IQ", "IT", "LY", "ML", "MR", "ME",
                  "MA", "NE", "NG", "MK", "PK", "PL", "RO", "RU", "SA", "SN", "RS", "SK", "SI", "SO", "ZA", "SS", "ES",
                  "SD", "CH", "SY", "TG", "TN", "TR", "UA", "UG", "ZM"):
        # print("ignoring country", cc, file=sys.stderr)
        continue
    for elem in doc.getElementsByTagName('*'):
        if elem.localName not in ["RDF", "Feature", "name", "officialName", "alternateName", "alt", "lat", "long",
                                  "parentCountry", "parentFeature"]:
            # print("getting rid of", elem)
            elem.parentNode.removeChild(elem)
    line = doc.toxml()
    if i != 0:
        idx = line.find("<gn:Feature")
        if idx != -1:
            line = line[idx:]
    line = line[:-len("</rdf:RDF>")]
    line = clean_dump_line(line=line)
    print(line)
    # print()
    i += 1

print("</rdf:RDF>")
'''
