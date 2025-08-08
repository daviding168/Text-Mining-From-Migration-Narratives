def load_seeds(lang):
    if lang == "en":
        userInput = [
            ["ROOT", -1,
             ["Accommodation", "Means of transport", "Environment", "Family Members"]],  # Label name only ####
            ["Accommodation", 0, ["home", "shelter"]],
            ["Means of transport", 0, ["boat", "train", "plane"]],
            ["Environment", 0, ["jungle", "island"]],
            ["Family Members", 0, ["mom", "relative", "daughter"]],
        ]
    elif lang == "fr":
        userInput = [
            ["ROOT", -1, ["Hébergement", "Moyens de transport", "Environnement", "Membres de famille"]],
            ["Hébergement", 0, ["appartement", "tente"]],
            ["Moyens de transport", 0, ["avion", "taxi", "métro"]],
            ["Environnement", 0, ["forêt", "mer"]],
            ["Membres de famille", 0, ["parents", "frères", "cousin"]]
        ]



    else:
        print("Please input a language!!!")

    return userInput


def annotated_taxonomy(lang):
    if lang == "en":
        userAnnotated = {

            "Accommodation": ["homes", "home", "house", "houses", "hostel", "room", "rooms", "hotel",
                              "apartment", "tent", "tents", "bed", "shelter", "floor", "garage", "building"],

            "Means of transport": ["foot", "walk", "autobus", "car", "cars", "taxi", "taxis",
                                   "cab", "bus", "buses", "minibus", "van", "vans", "minivan", "minivans",
                                   "trucks", "truck", "tractor", "trains", "train", "tram", "boat", "ship",
                                   "ferry", "plane", "airplane", "road", "flight"],

            "Environment": ["jungle", "jungles", "forest", "forests", "trees", "mountain", "mountains",
                            "desert", "hill", "valley", "rocks", "beach", "coast", "island", "islands", "river",
                            "sea", "seaside"],

            "Family Members": ["family", "father", "mother", "grandfather", "grandmother", "grandchildren", "parents",
                               "dad", "mum", "mom", "brother", "brothers", "sister", "sisters", "relatives", "relative",
                               "cousin", "cousins", "uncle", "wife", "husband", "kids", "child", "son", "sons",
                               "daughter", "daughters", "sister_in_law", "children", "friends", "friend"]
        }
    elif lang == "fr":
        userAnnotated = {

            "Hébergement": ["maison", "appartement", "chambre", "chambres", "maisonnette", "hôtel", "domicile", "foyer",
                            "tentes", "tente", "bâtiment", "bâtisse", "immeuble", "auberge", "cabane", "église",
                            "maisons", "ferme", "asile", "salle"],

            "Moyens de transport": ["bateau", "camion", "camions", "pickup", "marche", "pied", "transport_commun",
                                    "autobus", "train", "bus", "avion", "convoi", "véhicule", "véhicules", "taxi",
                                    "taxis", "minibus", "4x4", "autocar", "voiture", "voitures", "vol", "métro",
                                    "remorque", "fourgonnette", "route", "autoroute", "rail", "routes", "tram"],

            "Environnement": ["désert", "forêt", "forêts", "montagne", "montagnes", "brousse", "mer", "collines",
                              "rivière", "île", "jungle", "lac", "broussaille", "océan"],

            "Membres de famille": ["oncle", "oncles", "père", "frère", "grand-frère", "cousin", "cousins", "grand-père",
                                   "parent", "parents", "mère", "famille", "tante", "sœur", "sœurs", "maman", "frères",
                                   "papa", "tonton", "fils", "mari", "frangin", "cadet", "amie", "ami", "copain",
                                   "compatriote", "compatriotes", "fille", "copine", "camarade", "compagnon",
                                   "compagnons", "tuteur", "amis"]
        }

    else:
        print("Please input a language!!!")

    return userAnnotated
