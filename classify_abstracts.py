import json
import re
import numpy as np
import torch
import glob 
from sentence_transformers import SentenceTransformer
import argparse
from tech_classify import Model, LitModule
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('-xy', '--xy', dest='xy', type=str, help='Path to X,Y np array abstract_embs.npz')
parser.add_argument('-ck','--checkpoint', dest='ckpt', type=str, help='Path to pytorch-lightning model checkpoint')
parser.add_argument('-gd', '--graph_dict', dest='gd', type=str, help='Graph dict type')
parser.add_argument('-gp', '--graph_dict_path', dest='gp', type=str, help='Path to graph dict')
parser.add_argument('-ab', '--abs', dest='abs', type=str, help='Path to abstracts')
parser.add_argument('-out', '--out', dest='out', type=str, help='Path to output folder')
args = parser.parse_args()


FILTER_REGEX = ["World_War_", "[Hh]istor", "[Pp]eople", "^Ancient_", "_television_series", "^Industrial_", "_conferences$", "_museums$", "_companies$", "_organization", "_organisation", "_literature$", "_logos$", "_books$", "_art[s]*$", "_art[s]*\)$", "_painter[s]*$", "[Pp]ainting", "painting[s]*\)$", "^Art[s]*_", "[Aa]rtist", "_family[)]*$", "families", "officer", "[Mm]ember", "writer", "navigational_boxes$", "picture_set", "[Hh]otel", "\(music\)", "The_", "^\.", "_Revolution$", "academi", "educat", "_[Aa]ssociation[s]*$", "_Services$", "_neologisms", "^Filter_theory$", "[Ii]nvent", "_activists$", "_event[s]$", "_parties$", "Party", "_party$", "_comparison$", "_demos$", "_evangelism$", "_folklore$", "_movement$", "_terminology$", "_distributors$", "_failures$", "_ministries$", "_projects_", "_awards$", "_law$", "[Cc]ompan", "_program$", "_brand[s]$", "_website[s]*$", "^Websites_", "_incidents", "_regional_$", "_units$", "^Units_of_", "Discontinued_", "Technological_change", "Technological_pollution", "Technology_hazards", "Technology_transfer", "^Technology_[a-z]+$", "^Technophobia", "^Educational_", "wars$", "^Wars_", "^Esoteric_", "national", "ethnic", "food", "[Aa]nimal", "asteroids", "_venue", "wine", "_[mM]ission", "[Cc]ommission", "[Dd]rink", "[Aa]genc", "[tT]heory", "[Ll]anguage", "[tT]ournament", "[bB]attle", "[Mm]aps", "produce", "[Uu]nion", "[Tt]rade", "woodwork", "[nN]avy", "culture", "occupation", "[Cc]enter", "[rR]oyal", "Railway", "Defunct", "_Metro", "subsidiar", "[Aa]ward", "sausage", "East", "[Cc]orporat", "ethic", "accident", "forces", "locomotives", "[gG]roup", "disease", "disorder", "compet", "[Ff]oundation", "litigation", "acquisitions", "[Ee]xpedition", "[Cc]hallenge", "formations", "[cC]ommands", "Proposed", "Former", "installation", "[sS]hell", "[Ww]ar_", "[Nn]ame[ds]", "_bases", "[mM]urder", "[sS]uicide", "[cC]hampion", "vs", "accessories", "[Ss]ociet", "[Tt]reat(y|ies)", "_sites", "Map", "Sunday", "_Line", "terminals", "[tT]ax", "controvers", "Unsolved", "politic", "[Gg]arden", "Formal", "_regulation", "Tramway", "[Tt]rams", "[Rr]ailway", "[Tt]ests", "Test_", "fires", "projects", "EC_", "slang", "standards", "prefixes", "births", "[Dd]eath", "concept", "bridge", "basins", "UFO", "phenomena", "conspiracy", "Sports_car", "_teams", "funded", "_reaction[s]*$", "catalog", "Pira[tc][ey]", "_toys", "department", "Butch", "orbits", "special[it]", "coffee", "archaeol", "artifact", "practice", "real_estate", "saree", "Utopia", "humor", "areas", "[Gg]uerrilla", "legends", "[Ss]ociolo", "[eE]xam", "utilities", "_[pP]roducts", "Journal", "geograph", "Medieval", "[Ii]nfanticid", "dish", "castes", "[Ll]icense", "procedure", "cheese", "[Mm]ilk", "_bombings", "Proprietary", "disasters", "_firms", "studies", "_risk$", "_industry", "FAQ", "_marketing", "_instrument$", "_tools", "_equipment", "[_(]devices", "Non-profit", "[hH]ypothetical", "[Oo]bsolete", "specification", "_drugs", "intervals", "-mount_", "_safety$", "administration", "testing", "patterns", "bench", "appliance", "[dD]iagram", "utensil", "[Ff]ill", "[Pp]oint", "covers", "operations", "[Mm]erchand", "Servicio", "strateg", "[Aa]lliance", "certif", "[Ll]aborator", "[Ee]xperiment", "treatments", "service", "[Ff]raud", "counterfeit", "[Mm]easure", "tactics", "behavior", "Unit", "Old_World", "New_World", "Free_", "[Cc]omparison", "diet", "portal_selected", "[Vv]ulner", "[Oo]bservator", "[Pp]ark", "variables", "\.org", "[iI]mplement", "[Cc]ontainers", "[Cc]rossing", "National", "galler", "objects", "loaders", "[Oo]rganize", "interpreters", "_sets", "assets", "[bB]roken", "[cC]ollections", "hedge", "RNA", "[Qq]ualifications", "\.[a-zA-Z0_]", "fees", "_[sS]tone", "[Pp]roject$", "[Dd]uty", "Tango", "SS_", "[Ss]alv", "[pP]olic", "[Aa]irway", "[Dd]egree", "_[dD]ay[s]*$", "_[vV]accine", "_victims", "_[Rr]ule", "homicide", "[cC]rime", "[Rr]efugee", "[lL]eukemia", "trauma", "_[cC]ancer[s]*$", "_[Dd]rug[)s]*$", "[cC]hecklist", "_[sS]cheme", "[sS]core", "Play", "_[pP]rofile[s]*$", "_[Mm]ovement[s]*$", "[Pp]lant[)s]*$", "antagonist", "antibod", "[Dd]eficien", "Payment[)s]", "mm_", "\+", "antibio", "[Dd]ynast", "_protein[s]*$", "leader", "kinase", "_domain[s]*$", "[sS]ale", "activism", "activiti", "[Uu]prisi", "[Gg]overn", "_[eE]quat", "inhibitor", "_[Aa]ids$", "[rR]eich", "infrastructure", "_pricing", "explosion", "[-_]Express", "[0-9](th|st|nd|rd)", "[a-z][A-Z]"]

PREPOSITION_REGEX = ["_by_", "_in_", "_on_", "_for_", "_about_", "_during_", "_from_", "_at_", "_with_", "_along_", "_under_", "_de_", "_to_", "_who_", "_and_the_"]
FILTER_REGEX.extend(PREPOSITION_REGEX)

PLACE_NAME_REGEX = ["[iI]ndia", "[Uu]nited", "USA", "U\.S\.", "Chin", "chinese", "Sino", "[kK]orea", "[rR]ussia", "[fF]rench", "[gG]erman", "Spanish", "Spain", "Swiss", "Switzerland", "Portug", "[aA]ustr", "[mM]exic", "[aA]merica", "British", "Britain", "Brazil", "[eE]ngl", "[aA]frica", "Asia", "[eE]urope", "[cC]anad", "[aA]rgent", "[pP]akistan", "Ital", "Bangladesh", "Vietnam", "Indonesia", "[tT]hai", "Scot", "[iI]rish", "Ireland", "[cC]aliforn", "New_Zealand", "[Tt]urk", "[cC]olomb", "[nN]iger", "[pP]eru", "USSR", "Soviet", "[cC]zec", "Yugo", "Middle_East", "[Ii]ran", "Iraq", "Greek", "Greece", "Dutch", "[hH]olland", "Netherland", "[dD]anis", "[dD]enm", "Afghan", "[cC]ongo", "[aA]lban", "[Aa]lger", "[aA]ngol", "[Aa]nglo", "Antigua", "Armenia", "[aA]zerb", "[bB]ah", "Belarus", "[bB]elgi", "[bB]arb", "[bB]enin", "[bB]hutan", "[bB]osnia", "[bB]ulgaria", "[cC]ambod", "[cC]amero", "Cape", "[cC]had", "[cC]hile", "[cC]osta", "[cC]roat", "Cuba", "[dD]ji", "[dD]omini", "[eE]cua", "[eE]gypt", "El_", "[gG]uin", "[eE]stonia", "[fF][ui]j", "[fF]inl", "[fF]inn", "[gG]abo", "[gG]amb", "[gG]eorg", "[gG]han", "[sS]an[-_]", "[gG]uat", "[gG]uya", "[hH]ait", "[hH]ondu", "[hH]unga", "[Ii]cel", "[iI]sra]", "[iI]vory", "[jJ]amai", "[jJ]apa", "[Jj]ord", "[kK]az", "Kenya", "[kK]os", "[kK]uw", "Kyrgyzstan", "Latvia", "[lL]eban", "[lL]iberi", "Libya", "[lL]ith", "[mM]ala", "[mM]ald", "[mM]arsh", "[mM]aur", "Moldova", "[mM]onac", "Montenegro", "[mM]oro", "[mM]oz", "[mM]ya", "[bB]urm", "[nN]amib", "nN]icar", "[nN]orw", "[pP]ana", "[pP]hilippine", "[pP]olan", "Polish", "[qQ]at", "[rR]oman", "[rR]wan", "Saudi", "Arab", "Senegal", "Serbia", "Sierra", "Singapore", "Slovak", "Sloven", "Somal", "Lanka", "Tamil", "Suriname", "Sweden", "Swedish", "Syria", "Taiwan", "Trini", "Tunisia", "Tokyo", "Turk", "Ukrain", "Dubai", "Emirat", "UK", "States", "Urugu", "Uzbek", "Vatican", "Venezuela", "Yemen", "Zimbabwe", "Hong_Kong", "Berlin", "Paris", "London", "Boston", "[aA]rctic", "Antarctic", "Pacific", "Atlantic", "New_York", "Michigan", "Lake", "Manchest", "Manhattan", "Essex", "Hiroshima", "Normand", "Greenland", "Byzantin", "Long_Island", "Euro", "Bermuda", "Virgin", "Los_", "Eurasia", "USS", "Chelsea", "Celtic", "^US_", "US-", "Valencia", "Bristol", "Brussel", "[Bb]usa"]
FILTER_REGEX.extend(PLACE_NAME_REGEX)

TAXONOMY_REGEX = ["moth\)", "[Bb]utterfly", "genera$", "genus", "-class_", "classes", "classification", "Type_", "[Tt]axonom", "Edible", "_plants$", "_phyla", "ecozone", "flora", "fauna", "phylla", "phyllum", "nomenclature", "mammal", "greyhound", "[Hh]orse", "[Dd]olphin", "[Ww]hales", "bird[s)]", "species", "[vV]ulture", "[mM]onkey", "Strepto", "breed", "Fusar", "[pP]enic", "[Aa]llium", "Lasian", "[Cc]himpanz", "gorill", "_insect", "(_o|O)rders", "bacteria", "[Tt]ribe", "[a-z]ae$", "iale[s]*$", "mii", "dea$"]
FILTER_REGEX.extend(TAXONOMY_REGEX)

PERSON_NAME_REGEX = ["[mM]axwell", "[eE]instein", "Tesla", "Enid", "Winston", "[jJ]ung", "[dD]onald", "[mM]artin", "[Pp]hil", "[nN]i[kc]ol", "[lL]uigi", "[tT]homas", "[Hh]enr", "[Jj]oh", "[Mm]arc", "[Mm]ark(_|$)", "[Aa]brah", "Russ", "[gG]rah", "[mM]ona_", "[wW]ill", "McDonnell"]
FILTER_REGEX.extend(PERSON_NAME_REGEX)

CURRENCY_REGEX = ["_coin", "_dinar", "banknotes", "Dollar", "Rupee", "[fF]ranc", "Peso", "Dinar", "Ruppia", "Pound", "currency\)$"]
FILTER_REGEX.extend(CURRENCY_REGEX)

PLACE_TYPE_REGEX = ["_colleges$", "_universities$", "City", "[cC]ities", "[_(]city", "_[Pp]lace", "[vV]illage", "[Tt]own", "[Ee]mpire", "[cC]ounty", "counties", "[uU]niversit", "[cC]ollege", "[sS]chool", "[iI]nstitut", "_forest", "Forest", "stations$", "_housing", "shire", "[Mm]ountain", "[Bb]uildings", "structures", "[Cc]ommunit", "[Mm]unicipal", "[Dd]istrict", "land[s]*$", "facilit", "[Ii]land", "[Ii]sland", "River", "_river", "_road", "airport", "_camps", "plantation", "[vV]alley", "[Rr]esiden", "ondominiums", "[Ss]kyscraper", "[rR]estaurant", "locations", "Hemisphere", "[sS]hop", "zones", "[Nn]ations", "[cC]asino", r"[bB]ank[s]*\b", "hospital", "(_p|P)orts"]
FILTER_REGEX.extend(PLACE_TYPE_REGEX)

MUSIC_RELATED_REGEX = ["Ukulele", "[gG]uitar", "Piano", "[dD]rum", "labels", "musical", "_musicians$", "_music$", "Violin", "Cello", "Saxo", "[Bb]ass", "discograph", "soundtrack", "sounds", "recordi", "[Aa]lbum", "records", "_Records", "singles", "concert", "DJ", "bands", "[Cc]lub", "quartet", "genre[s]*[)]*$", "_festivals$", "_songs", "[Gg]haran", "[Bb]and\)", "Jazz", 'opera\)', "_Opera$", "Opera_"]
FILTER_REGEX.extend(MUSIC_RELATED_REGEX)

MEDIA_REGEX = ["Halo_", "BioShock", "Mario", "Sonic_hedgehog", "franchise\)$", "series\)$", "_media$", "_images$", "series_debuts$", "_series\)$", "_episodes", "_sitcoms", "[Vv]ideo_game", "[Cc]omic", "[Nn]ovel", "[Ff]iction", "_mod[s]*$", "[Ss]tudio", "[Ff]antasy", "_series$", "Works", "Book", "[_is]film", "_animation$", "_game[s]*$", "_literature$", "_magazines$", "_publication", "_journal[s]*$", "season", "[Tt]heater", "[Tt]heatre", "[bB]roadcasting", "videos", "channels", "Television", "manuals", "texts", "CD", "chapter", "manuscripts", "[eE]ntertainment", "[aA]nime", "[Mm]anga", "Mickey", "Marvel", "BBC", "PlayStation", "Nintendo", "Xbox", "Game_Boy", "Sega", "Superman", "Batman", "Ironman", "[dD]ocument", "show", "[Nn]ews", "film\)", "Internet_television", "TV", "Netflix", "gameplay"]
FILTER_REGEX.extend(MEDIA_REGEX)

RELIGION_REGEX = ["[cC]hurch", "[tT]emple", "Hindu", "Christ", "Juda", "Jain", "Buddh", "Islam", "[Jj]ew", "New_Age", "Religion[_-s]", "religion", "Religious_", "[Bb]ibl", "Catho", "[cC]rusade", "[Ss]aint", "Holy", "[sS]acre", "[Cc]hapel"]
FILTER_REGEX.extend(RELIGION_REGEX)

COMPUTER_RELATED_REGEX = ["RISC", "JavaScript", "Java", "Python", "KDE", "JSON", "C\+\+", "POSIX", "_[lL]ibrar", "Macintosh", "Mac[_]*OS", "Window[s]*", "OS_X", "\.NET", "PowerPC", "SPARC", "C_Sharp", "BASIC", "Pascal", "GNOME", "[uU]buntu", "X86", "Tizen", "OS$", "Torrent", "CONFIG", "Lisp", "_format", "[Pp]rotocol", "MeeGo", "MIPS", "Linux", "Unix", "Usenet", "_DVD", "[Bb]log", "Skype", "WhatsApp", "distributions", "Mega_Man", "[0-9]chan", "programs", "_Program", "[Bb]rowser", "_clients", "emulator", "OS/2", "hosting", "servers", "Open-source", "Haskell", "codec", "_processor", "Word", "debugger", "XMPP", "software", "Software_", "[fF]orum", "DVD[^_]", "Turing", "Framework", "RAID", "HTML", "MIME", "XML", "MP3", "MPEG", "LiveDistro"]
FILTER_REGEX.extend(COMPUTER_RELATED_REGEX)

SPORTS_REGEX = ["[sS]occer", "Olympic", "stadium", "[Ff]ootball", "[Bb]aseball", "[Bb]asketball", "[Cc]ricket", "[Tt]ennis", "[Bb]adminton", "hockey", "FIFA", "FC", "F.C.", "Wimbledon", "ICC", "NBC", "[eE]Sports", "Games", "_[sS]ports", "[sS]nowboard", "[Aa]rcher", "[Ss]ailors", "_racer", "racing", "[Cc]hess", "Go_", "Formula", "[aA]rcade", "Roller_coaster"]
FILTER_REGEX.extend(SPORTS_REGEX)

OCCUPATION_REGEX = ["[Cc]omed", "_designers$", "_broadcasters$", "_composers$", "_developers$", "_characters$", "_publishers$", "_ministers$", "_players$", "_personalities$", "_journalists$", "_winners$", "[cC]oach", "[Pp]hilosopher", "(_a|A)ctor", "[Ss]inger", "[dD]irector", "manager", "[eE]ngineers", "[dD]octor", "personnel", "[Aa]uthor", "[Rr]esearcher", "[Ww]omen", "pageant", "[Cc]orps", "editors", "[sS]cholar", "[Ff]ounder", "[Ww]orker", "[aA]rchitects", "[Pp]hysicist", "[Ee]mployee", "Driver", "Columbian", "commander", "_operators", "[aA]ssassin", "passenger", "provider", "makers", "[Pp]rofess", "attendant", "[Tt]raveler", "[cC]hemists", "mathematic", "[Ss]cientist", "drivers", "explorer", "aviator", "admiral", "[wW]indsurfer", "[Aa]rchiv", "[sS]culptor", "issue[rs]", "Actuaries", "[Ss]mith", "[Tt]heoris", "[Pp]ioneer", "[Hh]umanis", "[Pp]hysician", "_rider", "analyst", "insurgen", "advocate", "labour", "Florence", "_divers", "[Cc]onsultant", "retailers$", "_manufacturer", "engineer\)", "[sS]ponsor", "methodologis", "[Nn]urse", "ogist[s]*$", "admin", "_[aA]ct$", "survivor", "[Cc]horeo", "ist[s]*$", "[lL]eader", "inmate", "[Ss]ecreta", "[Ii]llustrat"]
FILTER_REGEX.extend(OCCUPATION_REGEX)

COMPANY_BRAND_REGEX = ["brand\)", "Microsoft", "Nvidia", "Amiga", "VIA", "TGV", "YouTube", "Yahoo", "Oracle", "Yamaha", "Nokia", "Honda", "Kawasaki", "Suzuki", "Nikon", "[Aa]pple", "Hewlett", "Autodesk", "Disney", "Twitter", "Sun_Micro", "Google", "[bB]erry", "Amazon", "IBM", "Lenovo", "Sony", "Ebay", "Credit_Suisse", "SAP", "Samsung", "Verizon", "Intel_", "Cisco", "Facebook", "Canon", "NEC", "UNESCO", "Magnavox", "Atari", "Solaris", "General_Motors", "General_Electric", "Steeldogs", "IPhone", "LG", "Ford", "Firefox", "Adobe", "ISO", "Panasonic", "Pentax", "Olympus", "Fuji", "Minolta", "Leica", "CBC", "Agfa", "Kodak", "Contax", "Cosina", "Mamiya", "Konica", "Yashica", "Rollei", "Voigtländer", "Merced", "Boeing", "Volks", "Toyota", "Hyundai", "Jaguar", "BMW", "Harley", "Audi_", "HP", "Bombardier", "UNIVAC", "DEC", "Lego", "IKEA", "WWE", "Nissan", "Mitsubishi", "Ferrari", "Samyang", "Renault", "Alfa_Romeo", "Fiat", "Lambor", "Maserati", "TVR", "Lotus", "Corvett", "Chevrolet", "Daim", "DÜWAG", "Zeiss", "Carl_Braun", "ARM", "Inc\.", "Xerox", "Dell", "McD", "ICL", "Acorn", "Elektronika", "Unisys", "Twitch", "Texas_Instruments", "Advanced_Micro_Devices", "Ricoh", "Sigma", "Hasselblad", "Casio", "Cyber-shot", "DiMAGE", "Sinclair", "SpongeBob", "Garfield", "Neo_Geo", "ISRO", "Vauxhall", "NSU", "NASA", "De_Dietrich", "Dolby", "Napier", "Rubik", "World_Health", "SpaceX", "Etihad", "IEEE", "Ducati", "Volvo", "Alstom", "Mozilla", "[kK]onami", "Huawei", "Qualcom", "CERN", "ARPA", "AT&T", "Airbus", "BenQ", "MediaTek", "CompuServe"]
FILTER_REGEX.extend(COMPANY_BRAND_REGEX)

filter_regexes = []
for s in FILTER_REGEX:
    filter_regexes.append(re.compile(s))


# Load classifier
def load_classifier(xy_path, ckpt_path):
    # This is bad code!!!
    data = np.load(xy_path)
    X_t, Y_t, X_n, Y_n = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
    num_samples = X_t.shape[0]

    X = np.concatenate((X_t, X_n[:num_samples]), axis=0)
    Y = np.concatenate((Y_t, Y_n[:num_samples]), axis=0)

    litmodel = LitModule.load_from_checkpoint(ckpt_path, input_dim=512, batch_size=32, X=X, Y=Y)
    litmodel.eval()
    model = litmodel.model
    return model

def load_abstracts(abs_path, graph_dict_type):
    fs = glob.glob(abs_path + '/' + graph_dict_type + '*')
    ent2abs = {}
    for path in tqdm(fs):
        with open(path, 'r', encoding='utf8') as f:
            js = json.load(f)
            for k in js:
                ent2abs[k] = js[k]
    return ent2abs

def run_tech_classifier(classifier, ent2abs, sent_model):
    ent2tech = {}
    for ent in tqdm(ent2abs):
        abstract = ent2abs[ent]
        if abstract == '':
            continue
        sent_emb = sent_model.encode(abstract)
        ten = torch.Tensor(sent_emb).to('cuda')
        logit = classifier(ten)
        if logit >= 0.5:
            ent2tech[ent] = 1
        else:
            ent2tech[ent] = 0
    return ent2tech

def run_regex_filters(filter_regexes, ent2abs):
    d = {}
    for name in ent2abs:
        entity = name.lower().replace("_"," ").replace("#","").replace("?","").replace(".","").capitalize()
        if any(regex.search(entity) for regex in filter_regexes):
            continue
        d[name] = ent2abs[name]
    return d


sent_model = SentenceTransformer('distiluse-base-multilingual-cased-v2')
sent_model.to('cuda')
print(next(sent_model.parameters()).is_cuda)


if __name__ == '__main__':
    # import pdb;pdb.set_trace()
    classifier = load_classifier(args.xy, args.ckpt)
    classifier.to('cuda')
    print(next(classifier.parameters()).is_cuda)
    print("#### Classifer loaded")
    ent2abs = load_abstracts(args.abs, args.gd)
    print("#### Abstracts loaded")
    print(len(ent2abs))
    newent2abs = run_regex_filters(filter_regexes, ent2abs)
    print("#### Regex loaded")
    print(len(newent2abs))
    ent2tech = run_tech_classifier(classifier, newent2abs, sent_model)

    with open(args.out + '/abstract_tech_' + args.gd + '.json', 'w', encoding = 'utf-8') as f:
        json.dump(ent2tech,f, ensure_ascii=False)    