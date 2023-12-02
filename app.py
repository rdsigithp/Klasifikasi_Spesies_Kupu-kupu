import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import streamlit as st
import requests
from PIL import Image
import numpy as np
from io import BytesIO

# Load the pre-trained model
model = load_model("MODEL_KLASIFIKASI_KUPU-KUPU")

# Fungsi untuk melakukan prediksi
def predict_species(img):
    # Praproses gambar
    img = img.resize((224, 224))  # Mengubah ukuran gambar
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    # Melakukan prediksi
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)

    # Get the class label
    labels = {
    0: '001.Atrophaneura_horishanus',
    1: '002.Atrophaneura_varuna',
    2: '003.Byasa_alcinous',
    3: '004.Byasa_dasarada',
    4: '005.Byasa_polyeuctes',
    5: '006.Graphium_agamemnon',
    6: '007.Graphium_cloanthus',
    7: '008.Graphium_sarpedon',
    8: '009.Iphiclides_podalirius',
    9: '010.Lamproptera_curius',
    10: '011.Lamproptera_meges',
    11: '012.Losaria_coon',
    12: '013.Meandrusa_payeni',
    13: '014.Meandrusa_sciron',
    14: '015.Pachliopta_aristolochiae',
    15: '016.Papilio_alcmenor',
    16: '017.Papilio_arcturus',
    17: '018.Papilio_bianor',
    18: '019.Papilio_dialis',
    19: '020.Papilio_hermosanus',
    20: '021.Papilio_hoppo',
    21: '022.Papilio_Krishna',
    22: '023.Papilio_maackii',
    23: '024.Papilio_machaon',
    24: '025.Papilio_memnon',
    25: '026.Papilio_nephelus',
    26: '027.Papilio_paris',
    27: '028.Papilio_polytes',
    28: '029.Papilio_prexaspes',
    29: '030.Papilio_protenor',
    30: '031.Papilio_xuthus',
    31: '032.Pathysa_antiphates',
    32: '033.Pazala_eurous',
    33: '034.Pazala_mullah',
    34: '035.Teinopalpus_aureus',
    35: '036.Teinopalpus_imperialis',
    36: '037.Troides_helena',
    37: '038.Troides_aeacus',
    38: '039.Troides_magellanus',
    39: '040.Bhutanitis_lidderdalii',
    40: '041.Luehdorfia_chinensis',
    41: '042.Sericinus_montelus',
    42: '043.Parnassius_apollo',
    43: '044.Parnassius_nomion',
    44: '045.Parnassius_phoebus',
    45: '046.Catopsilia_pomona',
    46: '047.Catopsilia_pyranthe',
    47: '048.Catopsilia_scylla',
    48: '049.Colias_erate',
    49: '050.Colias_fieldii',
    50: '051.Colias_hyale',
    51: '052.Colias_palaeno',
    52: '053.Eurema_blanda',
    53: '054.Eurema_andersoni',
    54: '055.Eurema_brigitta',
    55: '056.Eurema_hecabe',
    56: '057.Eurema_laeta',
    57: '058.Eurema_mandarina',
    58: '059.Gandaca_harina',
    59: '060.Gonepteryx_amintha',
    60: '061.Gonepteryx_rhamni',
    61: '062.Hebomoia_glaucippe',
    62: '063.Delias_acalis',
    63: '064.Delias_belladonna',
    64: '065.Delias_descombesi',
    65: '066.Delias_pasithoe',
    66: '067.Cepora_nerissa',
    67: '068.Pieris_canidia',
    68: '069.Pieris_melete',
    69: '070.Pieris_napi',
    70: '071.Pieris_rapae',
    71: '072.Leptidea_amurensis',
    72: '073.Leptidea_morsei',
    73: '074.Leptidea_sinapis',
    74: '075.Danaus_chrysippus',
    75: '076.Danaus_genutia',
    76: '077.Danaus_plexippus',
    77: '078.Euploea_core',
    78: '079.Euploea_midamus',
    79: '080.Euploea_sylvester',
    80: '081.Euploea_tulliolus',
    81: '082.Idea_leuconoe',
    82: '083.Ideopsis_similis',
    83: '084.Ideopsis_vulgaris',
    84: '085.Parantica_aglea',
    85: '086.Parantica_melaneus',
    86: '087.Parantica_sita',
    87: '088.Tirumala_limniace',
    88: '089.Tirumala_septentrionis',
    89: '090.Elymnias_hypermnestra',
    90: '091.Lethe_chandica',
    91: '092.Lethe_confusa',
    92: '093.Lethe_syrcis',
    93: '094.Mandarinia_regalis',
    94: '095.Melanitis_leda',
    95: '096.Melanitis_phedima',
    96: '097.Mycalesis_gotama',
    97: '098.Mycalesis_intermedia',
    98: '099.Mycalesis_perseus',
    99: '100.Neope_pulaha',
    100: '101.Penthema_darlisa',
    101: '102.Penthema_formosanum',
    102: '103.Ypthima_baldus',
    103: '104.Ypthima_praenubila',
    104: '105.Aemona_amathusia',
    105: '106.Faunis_eumeus',
    106: '107.Stichophthalma_howqua',
    107: '108.Apatura_ilia',
    108: '109.Apatura_iris',
    109: '110.Chitoria_ulupi',
    110: '111.Hestina_assimilis',
    111: '112.Rohana_parisatis',
    112: '113.Sasakia_charonda',
    113: '114.Sephisa_chandra',
    114: '115.Timelaea_albescens',
    115: '116.Ariadne_ariadne',
    116: '117.Ariadne_merione',
    117: '118.Euthalia_niepelti',
    118: '119.Athyma_perius',
    119: '120.Athyma_ranga',
    120: '121.Limenitis_sulpitia',
    121: '122.Neptis_hylas',
    122: '123.Neptis_miah',
    123: '124.Cyrestis_thyodamas',
    124: '125.Stibochiona_nicea',
    125: '126.Calinaga_buddha',
    126: '127.Clossiana_dia',
    127: '128.Clossiana_euphrosyne',
    128: '129.Clossiana_freija',
    129: '130.Clossiana_titania',
    130: '131.Cethosia_biblis',
    131: '132.Cethosia_cyane',
    132: '133.Damora_sagana',
    133: '134.Issoria_lathonia',
    134: '135.Proclossiana_eunomia',
    135: '136.Acraea_issoria',
    136: '137.Acraea_terpsicore',
    137: '138.Charaxes_bernardus',
    138: '139.Polyura_athamas',
    139: '140.Polyura_eudamippus',
    140: '141.Polyura_narcaea',
    141: '142.Doleschallia_bisaltide',
    142: '143.Hypolimnas_bolina',
    143: '144.Junonia_almana',
    144: '145.Junonia_atlites',
    145: '146.Junonia_hierta',
    146: '147.Junonia_iphita',
    147: '148.Junonia_orithya',
    148: '149.Kallima_inachus',
    149: '150.Kaniska_canace',
    150: '151.Polygonia_caureum',
    151: '152.Symbrenthia_lilaea',
    152: '153.Vanessa_cardui',
    153: '154.Vanessa_indica',
    154: '155.Libythea_myrrha',
    155: '156.Libythea_lepita',
    156: '157.Abisara_echerius',
    157: '158.Dodona_eugenes',
    158: '159.Zemeros_flegyas',
    159: '160.Allotinus_drumila',
    160: '161.Miletus_chinensis',
    161: '162.Taraka_hamada',
    162: '163.Curetis_acuta',
    163: '164.Amblopala_avidiena',
    164: '165.Arhopala_paramuta',
    165: '166.Arhopala_rama',
    166: '167.Artipe_eryx',
    167: '168.Horaga_albimacula',
    168: '169.Horaga_onyx',
    169: '170.Iraota_timoleon',
    170: '171.Jamides_bochus',
    171: '172.Lampides_boeticus',
    172: '173.Loxura_atymnus',
    173: '174.Mahathala_ameria',
    174: '175.Rapala_nissa',
    175: '176.SpinDasis_syama',
    176: '177.Tongeia_potanini',
    177: '178.Ussuriana_michaelis',
    178: '179.Zizeeria_maha',
    179: '180.Ampittia_virgata',
    180: '181.Ancistroides_nigrita',
    181: '182.Astictopterus_jama',
    182: '183.Erionota_torus',
    183: '184.Iambrix_salsala',
    184: '185.Isoteinon_lamprospilus',
    185: '186.Parnara_guttata',
    186: '187.Notocrypta_curvifascia',
    187: '188.Udaspes_folus',
    188: '189.Seseria_dohertyi',
    189: '190.Celaenorrhinus_maculosus',
    190: '191.Daimio_tethys',
    191: '192.Pseudocoladenia_dan',
    192: '193.Tagiades_menaka',
    193: '194.Abraximorpha_davidii',
    194: '195.Odontoptilum_angulatum',
    195: '196.Badamia_exclamationis',
    196: '197.Burara_gomata',
    197: '198.Hasora_anura',
    198: '199.Hasora_badra',
    199: '200.Hasora_vitta',
    }

    predicted_species = labels.get(predicted_class, 'Tidak Diketahui')

    # Mendapatkan probabilitas kelas yang diprediksi
    predicted_probability = predictions[0][predicted_class]

    return predicted_species, predicted_probability

# Streamlit UI
st.title("Klasifikasi Spesies Kupu-kupu")

# Option to upload image file
uploaded_file = st.file_uploader("Masukan Image kupu-kupu...", type=["jpg", "jpeg"])
if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)

    # Make predictions when the user clicks the button
    if st.button("Predict"):
        # Convert the uploaded file to Pillow Image
        img = Image.open(uploaded_file)
        predicted_species = predict_species(img)
        st.success(f"Termasuk Spesies Kupu-kupu: {predicted_species}")

# Option to upload image through URL
image_url = st.text_input("Masukan URL Image kupu-kupu:")
if image_url:
    try:
        # Fetch the image from the URL
        response = requests.get(image_url, stream=True)
        response.raise_for_status()

        # Open and display the image
        img = Image.open(BytesIO(response.content))
        st.image(img, caption="Uploaded Image.", use_column_width=True)

        # Make predictions when the user clicks the button
        if st.button("Predict"):
            predicted_species = predict_species(img)
            st.success(f"Termasuk Spesies Kupu-kupu: {predicted_species}")
    except Exception as e:
        st.error(f"Error: {e}")