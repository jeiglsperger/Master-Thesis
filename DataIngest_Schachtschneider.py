import pandas as pd
import numpy as np
from geopy.geocoders import Nominatim

def ingest_raw_data(data):
    file_df = pd.read_excel(data)

    file_df['Sorte'] = file_df['Sorte'].str.replace('.', '').str.replace('Ä', 'Ae').str.replace('ä', 'ae')\
        .str.replace('Ö', 'Oe').str.replace('ö', 'oe').str.replace('Ü', 'Ue').str.replace('ü', 'ue')\
        .str.replace('ß', 'ss').str.replace('NICHT', '').str.replace('in Sorten NICHT', '')\
        .str.replace('i.S. Nicht', '').str.replace(' A', '').str.replace('P 0,5', '').str.replace('P 1', '')\
        .str.replace(':', '').str.replace(' alt for P 1', '').str.replace(' x', '').str.replace('x ', '')\
        .str.replace(' C 3', '').str.replace(' C 4', '').str.replace('?', '').str.replace('?', '')\
        .str.replace(r'\d+', '').str.replace(' NGEBOT', '').str.replace(' N', '').str.replace(' Wi', '')\
        .str.replace(' Z', '').str.replace(' z', '').str.replace(' Fo', '').str.replace(' SA-', '')\
        .str.replace(' !!!', '').str.replace(' , guenstiger, da recht klei', '').str.replace(' ohne Bild', '')\
        .str.replace(' alt for', '').str.replace(' \)', '').str.replace(' Menge!!!', '').str.replace(' -', '')\
        .str.replace('- ', '').str.replace(' X', '').str.replace(' Ersatz', '').str.replace(' bereits g', '')\
        .str.replace('qmw', 'qm Zw').str.replace(' storno', '').str.replace(' lt Mitteilung Kunde, ha', '')\
        .str.replace(' oB', '').str.replace('§', '').str.replace(' oder', '').str.replace(' Haus', '')\
        .str.replace(' \( Stck haben gefehlt', '').str.replace(' Kisten', '').str.replace(' \(Stauden', '')\
        .str.replace(' a ', '').str.replace(',', '').str.replace(' o ', '').str.replace('  Kd Sander', '')\
        .str.replace(' micr ', ' microphylla ').str.replace('m²w', 'm² Zw')\
        .str.replace(' Menge nachfragen', '').str.replace(' SA b-ur', '').str.replace('\*\*\*', '')\
        .str.replace(' #', '').str.replace('  o', '').str.replace('®', '').str.replace('^.*Acaena.*$', 'Acaena')\
        .str.replace('^.*Acantholimon.*$', 'Acantholimon').str.replace('^.*Acanthus.*$', 'Acanthus')\
        .str.replace('^.*Achillea.*$', 'Achillea').str.replace('^.*Anemone.*$', 'Anemone')\
        .str.replace('^.*Achnatherum.*$', 'Achnatherum').str.replace('^.*Achillea.*$', 'Achillea')\
        .str.replace('^.*Yucca filamentosa.*$', 'Yucca filamentosa')\
        .str.replace('^.*Aconitum cammarum.*$', 'Aconitum cammarum')\
        .str.replace('^.*m² Stauden Silbersommer.*$', 'Silbersommer')\
        .str.replace('^.*m² Stauden Sommernachtstraum.*$', 'Sommernachtstraum')\
        .str.replace('^.*Blumenzwiebel.*$', 'Blumenzwiebeln').str.replace('^.*Aceriphyllum.*$', 'Aceriphyllum')\
        .str.replace('^.*qm Sommernachtstraum.*$', 'Sommernachtstraum')\
        .str.replace('^.*qm Stauden Silbersommer.*$', 'Silbersommer')\
        .str.replace('^.*Aconitum cammarum.*$', 'Aconitum cammarum')\
        .str.replace('^.*Yucca flaccida.*$', 'Yucca flaccida')\
        .str.replace('^.*kg Sedumsprossen.*$', 'kg Sedumsprossen').str.replace('^.*Jungpflanzen.*$', 'Jungpflanzen')\
        .str.replace('^.*Waldsteinia.*$', 'Waldsteinia').str.replace('^.*Aconitum.*$', 'Aconitum')\
        .str.replace('^.*Aconogonon sp.*$', 'Aconogonon speciosum')\
        .str.replace('^.*Acinos alpinus.*$', 'Acinos alpinus').str.replace('^.*Vinca major.*$', 'Vinca major')\
        .str.replace('^.*Vinca minor.*$', 'Vinca minor').str.replace('^.*Viola odorata.*$', 'Viola odorata')\
        .str.replace('^.*Achillea.*$', 'Achillea').str.replace('^.*Acinos alpinus.*$', 'Acinos alpinus')\
        .str.replace('^.*Veronicastrum virginicum.*$', 'Veronicastrum virginicum')\
        .str.replace('^.*Veronica spicata.*$', 'Veronica spicata').str.replace('^.*Acinos.*$', 'Acinos')\
        .str.replace('^.*Viola.*$', 'Viola').str.replace('^.*Acorus.*$', 'Acorus')\
        .str.replace('^.*Aconogonon.*$', 'Aconogonon').str.replace('^.*Verbena.*$', 'Verbena')\
        .str.replace('^.*Vernonia.*$', 'Vernonia').str.replace('^.*Achiella.*$', 'Achiella')\
        .str.replace('^.*Actaea.*$', 'Actaea').str.replace('^.*Veronica.*$', 'Veronica')\
        .str.replace('^.*Verbascum.*$', 'Verbascum').str.replace('^.*Iris.*$', 'Iris')\
        .str.replace('^.*Tulpia.*$', 'Tulipa').str.replace('^.*Kaiserkronen.*$', 'Kaiserkronen')\
        .str.replace('^.*Kalimeris.*$', 'Kalimeris').str.replace('^.*Jovibarba.*$', 'Jovibarba')\
        .str.replace('^.*Kniphofia.*$', 'Kniphofia').str.replace('^.*Kamille.*$', 'Kamille')\
        .str.replace('^.*Geranium.*$', 'Geranium').str.replace('^.*Equisetum.*$', 'Equisetum')\
        .str.replace('^.*Eremurus.*$', 'Eremurus').str.replace('^.*Delphinium.*$', 'Delphinium')\
        .str.replace('^.*Chrysanthemum.*$', 'Chrysanthemum').str.replace('^.*Bistorta.*$', 'Bistorta')\
        .str.replace('^.*Silene.*$', 'Silene').str.replace('^.*Thymus.*$', 'Thymus')\
        .str.replace('^.*Symphytum.*$', 'Symphytum').str.replace('^.*Tulipa.*$', 'Tulipa')\
        .str.replace('^.*Typha.*$', 'Typha').str.replace('^.*Salvia.*$', 'Salvia')\
        .str.replace('^.*Sambucus.*$', 'Sambucus').str.replace('^.*Polygonatum.*$', 'Polygonatum')\
        .str.replace('^.*Pennisetum.*$', 'Pennisetum').str.replace('^.*Penstemon.*$', 'Penstemon')\
        .str.replace('^.*Lupinus.*$', 'Lupinus').str.replace('^.*Allium.*$', 'Allium')\
        .str.replace('^.*Alstroemeria.*$', 'Alstroemeria').str.replace('^.*Alyssum.*$', 'Alyssum')\
        .str.replace('^.*Helleborus.*$', 'Helleborus').str.replace('^.*Hemerocallis.*$', 'Hemerocallis')\
        .str.replace('^.*Crocosmia.*$', 'Crocosmia').str.replace('^.*Cyclamen.*$', 'Cyclamen')\
        .str.replace('^.*Cotula.*$', 'Cotula').str.replace('^.*Alstroemeria.*$', 'Alstroemeria')\
        .str.replace('^.*Anchusa.*$', 'Anchusa').str.replace('^.*Sempervivum.*$', 'Sempervivum')\
        .str.replace('^.*Sidalcea.*$', 'Sidalcea').str.replace('^.*Erigeron.*$', 'Erigeron')\
        .str.replace('^.*Eryngium.*$', 'Eryngium').str.replace('^.*Erythronium.*$', 'Erythronium')\
        .str.replace('^.*Etiketten.*$', '').str.replace('^.*Cortaderia.*$', 'Cortaderia')\
        .str.replace('^.*Calamintha.*$', 'Calamintha').str.replace('^.*Uvularia.*$', 'Uvularia')\
        .str.replace('^.*Tradescantia.*$', 'Tradescantia').str.replace('^.*Tiarella.*$', 'Tiarella')\
        .str.replace('^.*Valeriana.*$', 'Valeriana').str.replace('^.*Buglossoides.*$', 'Buglossoides')\
        .str.replace('^.*Calamagrostis.*$', 'Calamagrostis').str.replace('^.*Butomus.*$', 'Butomus')\
        .str.replace('^.*Campanula.*$', 'Campanula').str.replace('^.*Sedum.*$', 'Sedum')\
        .str.replace('^.*Teucrium.*$', 'Teucrium').str.replace('^.*Thalictrum.*$', 'Thalictrum')\
        .str.replace('^.*Tiarella.*$', 'Tiarella').str.replace('^.*Stachys.*$', 'Stachys')\
        .str.replace('^.*Stipa.*$', 'Stipa').str.replace('^.*Phlodivaricata.*$', 'Phlodivaricata')\
        .str.replace('^.*aniculata.*$', 'Paniculata').str.replace('^.*Polystichum.*$', 'Polystichum')\
        .str.replace('^.*Agastache.*$', 'Agastache').str.replace('^.*Alcea.*$', 'Alcea')\
        .str.replace('^.*Aquilegia.*$', 'Aquilegia').str.replace('^.*Anthemis.*$', 'Anthemis')\
        .str.replace('^.*Armeria.*$', 'Armeria').str.replace('^.*Panicum.*$', 'Panicum')\
        .str.replace('^.*Papaver.*$', 'Papaver').str.replace('^.*Phalaris.*$', 'Phalaris')\
        .str.replace('^.*Heuchera.*$', 'Heuchera').str.replace('^.*Heucherella.*$', 'Heucherella')\
        .str.replace('^.*Hosta.*$', 'Hosta').str.replace('^.*Rudbeckia.*$', 'Rudbeckia')\
        .str.replace('^.*Sanguisorba.*$', 'Sanguisorba').str.replace('^.*Santolina.*$', 'Santolina')\
        .str.replace('^.*Saxifraga.*$', 'Saxifraga').str.replace('^.*Scabiosa.*$', 'Scabiosa')\
        .str.replace('^.*Scutellaria.*$', 'Scutellaria').str.replace('^.*Cynara.*$', 'Cynara')\
        .str.replace('^.*Dalina.*$', 'Dalina').str.replace('^.*Delosperma.*$', 'Delosperma')\
        .str.replace('^.*Dianthus.*$', 'Dianthus').str.replace('^.*Dicentra.*$', 'Dicentra')\
        .str.replace('^.*Dictamnus.*$', 'Dictamnus').str.replace('^.*Digiplexis.*$', 'Digiplexis')\
        .str.replace('^.*Doronicum.*$', 'Doronicum').str.replace('^.*Dryas.*$', 'Dryas')\
        .str.replace('^.*Dryopteris.*$', 'Dryopteris').str.replace('^.*Opuntia.*$', 'Opuntia')\
        .str.replace('^.*Origanum.*$', 'Origanum').str.replace('^.*Osmunda.*$', 'Osmunda')\
        .str.replace('^.*Oxalis.*$', 'Oxalis').str.replace('^.*Pachysandra.*$', 'Pachysandra')\
        .str.replace('^.*Paeonia.*$', 'Paeonia') .str.replace('^.*Aster.*$', 'Aster')\
        .str.replace('^.*Astilbe.*$', 'Astilbe').str.replace('^.*Clematis.*$', 'Clematis')\
        .str.replace('^.*Codonopsis.*$', 'Codonopsis').str.replace('^.*Colchicum.*$', 'Colchicum')\
        .str.replace('^.*Coleus.*$', 'Coleus').str.replace('^.*Colocasia.*$', 'Colocasia')\
        .str.replace('^.*Convallaria.*$', 'Convallaria').str.replace('^.*Convavulus.*$', 'Convavulus')\
        .str.replace('^.*Cordyline.*$', 'Cordyline').str.replace('^.*Coreopsis.*$', 'Coreopsis')\
        .str.replace('^.*Festuca.*$', 'Festuca').str.replace('^.*Filipendula.*$', 'Filipendula')\
        .str.replace('^.*Foeniculum.*$', 'Foeniculum').str.replace('^.*Heliopsis.*$', 'Heliopsis') \
        .str.replace('^.*Hepatica.*$', 'Hepatica').str.replace('^.*Herniaria.*$', 'Herniaria')\
        .str.replace('^.*Hesperis.*$', 'Hesperis').str.replace('^.*Hieracium.*$', 'Hieracium')\
        .str.replace('^.*Hippuris.*$', 'Hippuris').str.replace('^.*Holcus.*$', 'Holcus')\
        .str.replace('^.*Hottonia.*$', 'Hottonia').str.replace('^.*Houttuynia.*$', 'Houttuynia')\
        .str.replace('^.*Hutchinsia.*$', 'Hutchinsia').str.replace('^.*Hyacinthoides.*$', 'Hyacinthoides')\
        .str.replace('^.*Hyacinthoides.*$', 'Hyacinthoides').str.replace('^.*Hydrocharis.*$', 'Hydrocharis')\
        .str.replace('^.*Hydrocotyle.*$', 'Hydrocotyle').str.replace('^.*Hylomecon.*$', 'Hylomecon')\
        .str.replace('^.*Hypericum.*$', 'Hypericum').str.replace('^.*Bergenia.*$', 'Bergenia')\
        .str.replace('^.*Brunnera.*$', 'Brunnera').str.replace('^.*Boltonia.*$', 'Boltonia')\
        .str.replace('^.*Calceolaria.*$', 'Calceolaria').str.replace('^.*Caltha.*$', 'Caltha')\
        .str.replace('^.*Camassia.*$', 'Camassia').str.replace('^.*Cardiocrinum.*$', 'Cardiocrinum')\
        .str.replace('^.*Cardiocrinum.*$', 'Cardiocrinum').str.replace('^.*Stokesia.*$', 'Stokesia')\
        .str.replace('^.*Polemonium.*$', 'Polemonium').str.replace('^.*Pulsatilla.*$', 'Pulsatilla')\
        .str.replace('^.*Rosmarinus.*$', 'Rosmarinus').str.replace('^.*Euphorbia.*$', 'Euphorbia')\
        .str.replace('^.*Gaura.*$', 'Gaura').str.replace('^.*Geum.*$', 'Geum')\
        .str.replace('^.*Epimedium.*$', 'Epimedium').str.replace('^.*Erysimum.*$', 'Erysimum')\
        .str.replace('^.*Adenophora.*$', 'Adenophora').str.replace('^.*Adiantum.*$', 'Adiantum')\
        .str.replace('^.*Adonis.*$', 'Adonis').str.replace('^.*Fragaria.*$', 'Fragaria')\
        .str.replace('^.*Fritillaria.*$', 'Fritillaria').str.replace('^.*Fuchsia.*$', 'Fuchsia')\
        .str.replace('^.*Gaillardia.*$', 'Gaillardia').str.replace('^.*Galega.*$', 'Galega')\
        .str.replace('^.*Calla.*$', 'Calla').str.replace('^.*Callitriche.*$', 'Callitriche')\
        .str.replace('^.*Calystegia.*$', 'Calystegia').str.replace('^.*Canna.*$', 'Canna')\
        .str.replace('^.*Cardamine.*$', 'Cardamine').str.replace('^.*Care.*$', 'Carex')\
        .str.replace('^.*Centaurea.*$', 'Centaurea').str.replace('^.*Centranthus.*$', 'Centranthus')\
        .str.replace('^.*Gentiana.*$', 'Gentiana').str.replace('^.*Glechoma.*$', 'Glechoma')\
        .str.replace('^.*Gunnera.*$', 'Gunnera').str.replace('^.*Stratiotes.*$', 'Stratiotes')\
        .str.replace('^.*Sunsatia.*$', 'Sunsatia').str.replace('^.*Tagetes.*$', 'Tagetes')\
        .str.replace('^.*Tanacetum.*$', 'Tanacetum').str.replace('^.*Tellima.*$', 'Tellima')\
        .str.replace('^.*Trapa.*$', 'Trapa').str.replace('^.*Tricyrtis.*$', 'Tricyrtis')\
        .str.replace('^.*Trillium.*$', 'Trillium').str.replace('^.*Trollius.*$', 'Trollius')\
        .str.replace('^.*Primula.*$', 'Primula').str.replace('^.*Pulmonaria.*$', 'Pulmonaria')\
        .str.replace('^.*Rheum.*$', 'Rheum').str.replace('^.*Rodgersia.*$', 'Rodgersia')\
        .str.replace('^.*Rosa.*$', 'Rosa').str.replace('^.*Sagina.*$', 'Sagina')\
        .str.replace('^.*Saponaria.*$', 'Saponaria').str.replace('^.*Rodgersia.*$', 'Rodgersia')\
        .str.replace('^.*Rhodoxis.*$', 'Rhodoxis').str.replace('^.*Perovskia.*$', 'Perovskia')\
        .str.replace('^.*Agapanthus.*$', 'Agapanthus').str.replace('^.*Chaenarrhinum.*$', 'Chaenarrhinum')\
        .str.replace('^.*Cerastium.*$', 'Cerastium').str.replace('^.*Carum.*$', 'Carum')\
        .str.replace('^.*Caryopteris.*$', 'Caryopteris').str.replace('^.*Catananche.*$', 'Catananche')\
        .str.replace('^.*Caunassia.*$', 'Caunassia').str.replace('^.*Centaurium.*$', 'Centaurium')\
        .str.replace('^.*Cephalanthera.*$', 'Cephalanthera').str.replace('^.*Cephalaria.*$', 'Cephalaria')\
        .str.replace('^.*Ceratophyllum.*$', 'Ceratophyllum').str.replace('^.*Ceratostigma.*$', 'Ceratostigma')\
        .str.replace('^.*Chaenar.hinum.*$', 'Chaenarrhinum').str.replace('^.*Chamaemelum.*$', 'Chamaemelum')\
        .str.replace('^.*Chasmanthium.*$', 'Chasmanthium').str.replace('^.*Chel.donium.*$', 'Chelidonium')\
        .str.replace('^.*Chelone.*$', 'Chelone')\
        .str.replace('^.*Chiastophyllum oppos.*$', 'Chiastophyllum oppositifolium')\
        .str.replace('^.*Chrys.*$', 'Chrysanthemum').str.replace('^.*Chrysogonum.*$', 'Chrysogonum')\
        .str.replace('^.*Chrysopogon.*$', 'Chrysopogon').str.replace('^.*Chrysopsis.*$', 'Chrysopsis')\
        .str.replace('^.*Cichorium.*$', 'Cichorium').str.replace('^.*Cimicifuga.*$', 'Cimicifuga')\
        .str.replace('^.*Cistus.*$', 'Cistus').str.replace('^.*Lilium.*$', 'Lilium')\
        .str.replace('^.*Leucanthemum.*$', 'Leucanthemum').str.replace('^.*Leucojum.*$', 'Leucojum')\
        .str.replace('^.*Leucqanthemum.*$', 'Leucanthemum').str.replace('^.*Levisticum.*$', 'Levisticum')\
        .str.replace('^.*Leymus.*$', 'Leymus').str.replace('^.*Liatris.*$', 'Liatris')\
        .str.replace('^.*Lili.*$', 'Lilium').str.replace('^.*Hakonechloa.*$', 'Hakonechloa')\
        .str.replace('^.*Hedera.*$', 'Hedera').str.replace('^.*Helenium.*$', 'Helenium')\
        .str.replace('^.*Helianthemum.*$', 'Helianthemum').str.replace('^.*Helianthus.*$', 'Helianthus')\
        .str.replace('^.*Gypsophila.*$', 'Gypsophila').str.replace('^.*Hakonechloa.*$', 'Hakonechloa')\
        .str.replace('^.*Nepeta.*$', 'Nepeta').str.replace('^.*Nuphar.*$', 'Nuphar')\
        .str.replace('^.*Nymphaea.*$', 'Nymphaea').str.replace('^.*Primula.*$', 'Primula')\
        .str.replace('^.*Prunella.*$', 'Prunella').str.replace('^.*Rheum.*$', 'Rheum')\
        .str.replace('^.*Pycnanthemum.*$', 'Pycnanthemum').str.replace('^.*Rhodohypoxis.*$', 'Rhodohypoxis')\
        .str.replace('^.*Rhodoxis.*$', 'Rhodoxis').str.replace('^.*Rodgersia.*$', 'Rodgersia')\
        .str.replace('^.*Rosa.*$', 'Rosa').str.replace('^.*Sagina.*$', 'Sagina')\
        .str.replace('^.*Saponaria.*$', 'Saponaria').str.replace('^.*Tanacetum.*$', 'Tanacetum')\
        .str.replace('^.*Tellima.*$', 'Tellima').str.replace('^.*Telekia.*$', 'Telekia')\
        .str.replace('^.*Tricyrtis.*$', 'Tricyrtis').str.replace('^.*Trifolium.*$', 'Trifolium')\
        .str.replace('^.*Trillium.*$', 'Trillium').str.replace('^.*Trollius.*$', 'Trollius')\
        .str.replace('^.*Cosmos.*$', 'Cosmos').str.replace('^.*Cotoneaster.*$', 'Cotoneaster')\
        .str.replace('^.*Crambe.*$', 'Crambe').str.replace('^.*Crassula.*$', 'Crassula')\
        .str.replace('^.*Crocus.*$', 'Crocus').str.replace('^.*Cymbalaria.*$', 'Cymbalaria')\
        .str.replace('^.*Cymbopogon.*$', 'Cymbopogon').str.replace('^.*Cynoglossum.*$', 'Cynoglossum')\
        .str.replace('^.*Cyperus.*$', 'Cyperus').str.replace('^.*Cypripedium.*$', 'Cypripedium')\
        .str.replace('^.*Cyrtomium.*$', 'Cyrtomium').str.replace('^.*Dach.*$', 'Dachgartenstauden')\
        .str.replace('^.*Dactylorhiza.*$', 'Dactylorhiza').str.replace('^.*Dahlia.*$', 'Dahlia')\
        .str.replace('^.*Daphne.*$', 'Daphne').str.replace('^.*Darmera.*$', 'Darmera')\
        .str.replace('^.*Dendran.*$', 'Dendranthema').str.replace('^.*Lavandula.*$', 'Lavandula')\
        .str.replace('^.*Lavatera.*$', 'Lavatera').str.replace('^.*Miscanthus.*$', 'Miscanthus')\
        .str.replace('^.*Lonicera.*$', 'Lonicera').str.replace('^.*Lotos.*$', 'Lotos')\
        .str.replace('^.*Lunaria.*$', 'Lunaria').str.replace('^.*Luzula.*$', 'Luzula')\
        .str.replace('^.*Lychnis.*$', 'Lychnis').str.replace('^.*Lysichiton.*$', 'Lysichiton')\
        .str.replace('^.*Lysimachia.*$', 'Lysimachia').str.replace('^.*Lythrum.*$', 'Lythrum')\
        .str.replace('^.*Macleaya.*$', 'Macleaya').str.replace('^.*Major.*$', 'Majoran')\
        .str.replace('^.*Malva.*$', 'Malva').str.replace('^.*Mandragona.*$', 'Mandragona')\
        .str.replace('^.*Margarit.*$', 'Margarite').str.replace('^.*Marrubium.*$', 'Marrubium')\
        .str.replace('^.*Matricaria.*$', 'Matricaria').str.replace('^.*Matteuccia.*$', 'Matteuccia')\
        .str.replace('^.*Vinca.*$', 'Vinca').str.replace('^.*Phloarendsii.*$', 'Phloarendsii')\
        .str.replace('^.*Phlosubulata.*$', 'Phlosubulata').str.replace('^.*Oenothera.*$', 'Oenothera')\
        .str.replace('^.*Omphalodes.*$', 'Omphalodes').str.replace('^.*Ophiopogon.*$', 'Ophiopogon')\
        .str.replace('^.*Persicaria.*$', 'Persicaria').str.replace('^.*Petroselinum.*$', 'Petroselinum')\
        .str.replace('^.*Deschampsia.*$', 'Deschampsia').str.replace('^.*Diascia.*$', 'Diascia')\
        .str.replace('^.*Digitalis.*$', 'Digitalis').str.replace('^.*Dipsacus.*$', 'Dipsacus')\
        .str.replace('^.*Dodecatheon.*$', 'Dodecatheon').str.replace('^.*Draba.*$', 'Draba')\
        .str.replace('^.*Echinacea.*$', 'Echinacea').str.replace('^.*Dracocephalum.*$', 'Dracocephalum')\
        .str.replace('^.*Lotus.*$', 'Lotus').str.replace('^.*Maianthemum.*$', 'Maianthemum')\
        .str.replace('^.*Mentha.*$', 'Mentha').str.replace('^.*Mertensia.*$', 'Mertensia')\
        .str.replace('^.*Milium.*$', 'Milium').str.replace('^.*Mimulus.*$', 'Mimulus')\
        .str.replace('^.*Polygonum.*$', 'Polygonum').str.replace('^.*Potentilla.*$', 'Potentilla')\
        .str.replace('^.*Pontederia.*$', 'Pontederia').str.replace('^.*Deschampsia.*$', 'Deschampsia')\
        .str.replace('^.*Echinacea.*$', 'Echinacea').str.replace('^.*Monarda.*$', 'Monarda')\
        .str.replace('^.*Myosotis.*$', 'Myosotis').str.replace('^.*Myriophyllum.*$', 'Myriophyllum')\
        .str.replace('^.*Myrrhis.*$', 'Myrrhis').str.replace('^.*Narcissus.*$', 'Narcissus')\
        .str.replace('^.*Neopaxia.*$', 'Neopaxia').str.replace('^.*Onoclea.*$', 'Onoclea')\
        .str.replace('^.*Solidago.*$', 'Solidago').str.replace('^.*Stauden.*$', 'Stauden')\
        .str.replace('^.*Ajuga.*$', 'Ajuga').str.replace('^.*Alchemilla.*$', 'Alchemilla')\
        .str.replace('^.*Amsonia.*$', 'Amsonia').str.replace('^.*Iberis.*$', 'Iberis')\
        .str.replace('^.*Liriope.*$', 'Liriope').str.replace('^.*Eichhornia.*$', 'Eichhornia')\
        .str.replace('^.*Eleocharis.*$', 'Eleocharis').str.replace('^.*Epilobium.*$', 'Epilobium')\
        .str.replace('^.*Erodium.*$', 'Erodium').str.replace('^.*Eucalyptus.*$', 'Eucalyptus')\
        .str.replace('^.*Echinops.*$', 'Echinops').str.replace('^.*Eupatorium.*$', 'Eupatorium')\
        .str.replace('^.*Fallopia.*$', 'Fallopia').str.replace('^.*Farfugium.*$', 'Farfugium')\
        .str.replace('^.*Fargesia.*$', 'Fargesia').str.replace('^.*Farn.*$', 'Farn')\
        .str.replace('^.*Felicia.*$', 'Felicia').str.replace('^.*Fenchel.*$', 'Fenchel')\
        .str.replace('^.*Fritt.*$', 'Frittelaria').str.replace('^.*Frttilaria.*$', 'Frttilaria')\
        .str.replace('^.*Helichrysum.*$', 'Helichrysum').str.replace('^.*Hibiscus.*$', 'Hibiscus')\
        .str.replace('^.*Hyssopus.*$', 'Hyssopus').str.replace('^.*Astrantia.*$', 'Astrantia')\
        .str.replace('^.*Corydalis.*$', 'Corydalis').str.replace('^.*Cosmea.*$', 'Cosmea')\
        .str.replace('^.*Hasenschwanzgras.*$', 'Hasenschwanzgras')\
        .str.replace('^.*Helictotrichon.*$', 'Helictotrichon').str.replace('^.*Humulus.*$', 'Humulus')\
        .str.replace('^.*Hystripatula.*$', 'Hystripatula').str.replace('^.*Imperata.*$', 'Imperata')\
        .str.replace('^.*Incarvillea.*$', 'Incarvillea').str.replace('^.*Klatschmohn.*$', 'Klatschmohn')\
        .str.replace('^.*Lamiastrum.*$', 'Lamium').str.replace('^.*Lamium.*$', 'Lamium')\
        .str.replace('^.*Lathyrus.*$', 'Lathyrus').str.replace('^.*Knautia.*$', 'Knautia')\
        .str.replace('^.*Lemna.*$', 'Lemna').str.replace('^.*Leontopodium.*$', 'Leontopodium')\
        .str.replace('^.*Lewisia.*$', 'Lewisia').str.replace('^.*Ligularia.*$', 'Ligularia')\
        .str.replace('^.*Pleione.*$', 'Pleione').str.replace('^.*Plumbago.*$', 'Plumbago')\
        .str.replace('^.*Polianth.*$', 'Polianthes').str.replace('^.*Potamogeton.*$', 'Potamogeton')\
        .str.replace('^.*Pratia.*$', 'Pratia').str.replace('^.*Pritzelago.*$', 'Pritzelago')\
        .str.replace('^.*Koeleria.*$', 'Koeleria').str.replace('^.*Linum.*$', 'Linum')\
        .str.replace('^.*Linaria.*$', 'Linaria').str.replace('^.*Limonium.*$', 'Limonium')\
        .str.replace('^.*Lithodora.*$', 'Lithodora').str.replace('^.*Ranunculus.*$', 'Ranunculus')\
        .str.replace('^.*Scirpus.*$', 'Scirpus').str.replace('^.*Sisyrinchium.*$', 'Sisyrinchium')\
        .str.replace('^.*Sesleria.*$', 'Sesleria').str.replace('^.*Aegopodium.*$', 'Aegopodium')\
        .str.replace('^.*Aethionema.*$', 'Aethionema').str.replace('^.*Agapanthus.*$', 'Agapanthus')\
        .str.replace('^.*Agrimonia.*$', 'Agrimonia').str.replace('^.*Agrostemma.*$', 'Agrostemma')\
        .str.replace('^.*Agrostis.*$', 'Agrostis').str.replace('^.*Ajania.*$', 'Ajania')\
        .str.replace('^.*Koriander.*$', 'Koriander').str.replace('^.*Hydrangea.*$', 'Hydrangea')\
        .str.replace('^.*Molinia.*$', 'Molinia').str.replace('^.*Muscari.*$', 'Muscari')\
        .str.replace('^.*Phlo.*$', 'Phlox').str.replace('^.*Acer palmatum (U).*$', 'Acer palmatum')\
        .str.replace('^.*Aethion.*$', 'Aethionema').str.replace('^.*Agapanthus.*$', 'Agapanthus')\
        .str.replace('^.*Akebia.*$', 'Akebia').str.replace('^.*Alcalthaea.*$', 'Alcalthaea')\
        .str.replace('^.*Alisma.*$', 'Alisma').str.replace('^.*Alliaria.*$', 'Alliaria')\
        .str.replace('^.*Alliu.*$', 'Allium').str.replace('^.*Alium.*$', 'Allium')\
        .str.replace('^.*Alopecurus.*$', 'Alopecurus').str.replace('^.*Aloysia.*', 'Aloysia')\
        .str.replace('^.*Alstro.*$', 'Alstroemeria').str.replace('^.*Anacyclus.*$', 'Anacyclus')\
        .str.replace('^.*Ammophila.*$', 'Ammophila').str.replace('^.*Alth.*$', 'Althaea')\
        .str.replace('^.*Anagal.*$', 'Anagallis').str.replace('^.*Ananas Salbei.*$', 'Ananassalbei')\
        .str.replace('^.*Ananas-Minze.*$', 'Ananasminze').str.replace('^.*Ananassalvia.*$', 'Ananassalbei')\
        .str.replace('^.*Anaphalis.*$', 'Anaphalis').str.replace('^.*Andropogon.*$', 'Andropogon')\
        .str.replace('^.*Androsace.*$', 'Androsace').str.replace('^.*Anemmone.*$', 'Anemone')\
        .str.replace('^.*Anemonopsis.*$', 'Anemonopsis').str.replace('^.*Anethum.*$', 'Anethum')\
        .str.replace('^.*Angelica.*$', 'Angelica').str.replace('^.*Antennaria.*$', 'Antennaria')\
        .str.replace('^.*Anthericum.*$', 'Anthericum').str.replace('^.*Anthriscus.*$', 'Anthriscus')\
        .str.replace('^.*Anthyllis.*$', 'Anthyllis').str.replace('^.*Antirrhin.*$', 'Antirrhinum')\
        .str.replace('^.*Aponogeton.*$', 'Aponogeton').str.replace('^.*Arabis.*$', 'Arabis')\
        .str.replace('^.*Aralia.*$', 'Aralia').str.replace('^.*Arctanthemum.*$', 'Arctanthemum')\
        .str.replace('^.*Arctostaphylos.*$', 'Arctostaphylos').str.replace('^.*Arenaria.*$', 'Arenaria')\
        .str.replace('^.*Arisaema.*$', 'Arisaema').str.replace('^.*Aristolochia.*$', 'Aristolochia')\
        .str.replace('^.*Armoracia.*$', 'Armoracia').str.replace('^.*Aubrieta.*$', 'Aubrieta')\
        .str.replace('^.*Arnica.*$', 'Arnica').str.replace('^.*Arrhenatherum.*$', 'Arrhenatherum')\
        .str.replace('^.*Artemisia.*$', 'Artemisia').str.replace('^.*Arum.*$', 'Arum')\
        .str.replace('^.*Aruncus.*$', 'Aruncus').str.replace('^.*Arundo.*$', 'Arundo')\
        .str.replace('^.*Asarum.*$', 'Asarum').str.replace('^.*Asclepia.*$', 'Asclepia')\
        .str.replace('^.*Asparagus.*$', 'Asparagus').str.replace('^.*Asperula.*$', 'Asperula')\
        .str.replace('^.*Asphodeline.*$', 'Asphodeline').str.replace('^.*Asphodelus.*$', 'Asphodelus')\
        .str.replace('^.*Asplen.*$', 'Asplenium').str.replace('^.*Astilboides.*$', 'Astilboides')\
        .str.replace('^.*Athyrium.*$', 'Athyrium').str.replace('^.*Atropa.*$', 'Atropa')\
        .str.replace('^.*Avena.*$', 'Avena').str.replace('^.*Azolla.*$', 'Azolla')\
        .str.replace('^.*Azorella.*$', 'Azorella').str.replace('^.*Bacopa.*$', 'Bacopa')\
        .str.replace('^.*Ballota.*$', 'Ballota').str.replace('^.*Bambus.*$', 'Bambus')\
        .str.replace('^.*Baptisia.*$', 'Baptisia').str.replace('^.*Barbarea.*$', 'Barbarea')\
        .str.replace('^.*Basilikum.*$', 'Basilikum').str.replace('^.*Beetrose.*$', 'Beetrose')\
        .str.replace('^.*Begonia.*$', 'Begonia').str.replace('^.*Berberis.*$', 'Berberis')\
        .str.replace('^.*Bergpolster.*$', 'Bergpolster').str.replace('^.*Bergprimel.*$', 'Bergprimel')\
        .str.replace('^.*Beta.*$', 'Beta').str.replace('^.*Birne .*$', 'Birne')\
        .str.replace('^.*Blechnum.*$', 'Blechnum').str.replace('^.*Bletilla.*$', 'Bletilla')\
        .str.replace('^.*Borago.*$', 'Borago').str.replace('^.*Borretsch.*$', 'Borago')\
        .str.replace('^.*Bouteloua.*$', 'Bouteloua').str.replace('^.*Brassica.*$', 'Brassica')\
        .str.replace('^.*Briza.*$', 'Briza').str.replace('^.*Brombeere.*$', 'Brombeere')\
        .str.replace('^.*Buddle.*$', 'Buddleja').str.replace('^.*Bupht.*$', 'Buphthalmum')\
        .str.replace('^.*Buxus.*$', 'Buxus')

    file_df['Sorte'] = file_df['Sorte'].str.strip()

    file_df = file_df[(file_df.Sorte != 'sonstige') & (file_df.Sorte != 'verauslagte Portokosten')
                      & (file_df.Sorte != 'erhaltenenzahlung Re  vom')
                      & (file_df.Sorte != 'geleistetenzahlung aus Re-Nr')
                      & (file_df.Sorte != 'zusaetzlicher Bonus fuer') & (file_df.Sorte != 'keine!    Uebergabe auf IPM')
                      & (file_df.Sorte != 'Toepfe fuer Schwimmpflanzen') & (file_df.Sorte != '^.*Fracht.*$')
                      & (file_df.Sorte != '^.*Werbematerial.*$') & (file_df.Sorte != '^.*Vita.Verde.*$')
                      & (file_df.Sorte != '^.*Europalette.*$') & (file_df.Sorte != '^.*Faltbla.*$')
                      & (file_df.Sorte != '^.*Fracht.*$') & (file_df.Sorte != '^.*Gartenschaetze.*$')
                      & (file_df.Sorte != '^.*Gartenschaetze.*$') & (file_df.Sorte != '^.*Holzkisten geliefert.*$')
                      & (file_df.Sorte != '^.*Holzkisten geliefert.*$')
                      & (file_df.Sorte != '^.*Infomaterialktionspakete.*$')
                      & (file_df.Sorte != '^.*Prospektstaender.*$') & (file_df.Sorte != '^.*Holzkisten geliefert.*$')
                      & (file_df.Sorte != '^.*Aloe Vera  haben wir nicht im Sortiment.*$')
                      & (file_df.Sorte != '^.*Amsolis  sind uns leider nicht bekanntAuch nicht.*$')
                      & (file_df.Sorte != '^.*Angebotspaket.*$') & (file_df.Sorte != '^.*zahlung.*$')
                      & (file_df.Sorte != '^.*Anzeigenvorlage Planten ut`norden.*$')
                      & (file_df.Sorte != '^.*Aufkleber fuer Saeule.*$') & (file_df.Sorte != '^.*Aufsetzer.*$')
                      & (file_df.Sorte != '^.*Ausstellen der Pflanzen in die Beete  pauschal.*$')
                      & (file_df.Sorte != '^.*Ausstellungsflaeche.*$') & (file_df.Sorte != '^.*Auswaschen.*$')
                      & (file_df.Sorte != 'Bestellblaetter') & (file_df.Sorte != 'Bestellformular')
                      & (file_df.Sorte != '^.*Bild.*$') & (file_df.Sorte != '^.*etikette.*$')
                      & (file_df.Sorte != '^.*Broschueren.*$')]

    file_df['Einzelpreis'] = file_df['Einzelpreis'].replace('€', '', regex=True).replace(',', '.', regex=True)

    file_df['Gesamtpreis'] = file_df['Einzelpreis']*file_df['Menge']
    file_df['Gesamtpreis'] = file_df['Gesamtpreis'].round(2)

    file_df = file_df.filter(['Gesamtpreis', 'Sorte', 'Auf. Datum']).dropna()

    file_df['Auf. Datum'] = pd.to_datetime(file_df['Auf. Datum'], format='%d.%m.%Y')
    file_df['Auf. Datum'] = file_df['Auf. Datum'].dt.date

    file_df = file_df.sort_values(by=['Auf. Datum'])
    file_df = file_df.groupby(['Auf. Datum', 'Sorte'], as_index=False).sum()

    return file_df


sum_df_2001_2007 = ingest_raw_data("Rechnungsübersicht_2001-2007.xlsx")
sum_df_2008_2013 = ingest_raw_data("Rechnungsübersicht 2008-2013.xlsx")
sum_df_2014_2018 = ingest_raw_data("Rechnungsübersicht 2014-2018.xlsx")
sum_df_2019_2020 = ingest_raw_data("Rechnungsübersicht 2019-2020.xlsx")

df = pd.concat([sum_df_2001_2007, sum_df_2008_2013, sum_df_2014_2018, sum_df_2019_2020])
sum_df = df.pivot_table(values='Gesamtpreis', index='Auf. Datum', columns='Sorte', fill_value=0, aggfunc=np.sum)
sum_df = sum_df.loc[:, (sum_df != 0).any(axis=0)]
sum_df.to_csv("Schachtschneider_new.csv")

"""
The ingestion was splitted into two scripts to reduce computation time. The following is the second script. For better 
overview in put both in one script.
"""

file_df = pd.read_csv("Schachtschneider_further.csv")

file_df = file_df.set_index('Auf. Datum')

file_df.columns = file_df.columns.str.replace('^.*Carlina.*$', 'Carlina').str.replace('^.*Cirsium.*$', 'Cirsium')\
    .str.replace('^.*Alca.*$', 'Alcalthaea').str.replace('^.*Calamagrostis.*$', 'Calamagrostis')\
    .str.replace('^.*Calendula.*$', 'Calendula').str.replace('^.*Chenopodium.*$', 'Chenopodium')\
    .str.replace('^.*Chionodoxa.*$', 'Chionodoxa').str.replace('^.*Chryanthemum.*$', 'Chrysanthemum')\
    .str.replace('^.*Cimifuga.*$', 'Cimicifuga').str.replace('^.*Colcicum.*$', 'Colchicum')\
    .str.replace('^.*Convolovus.*$', 'Convolvulus').str.replace('^.*Coriandrum.*$', 'Coriandrum')\
    .str.replace('^.*Cornus.*$', 'Cornus').str.replace('^.*Coryllus.*$', 'Coryllus')\
    .str.replace('^.*Cotoneatser.*$', 'Cotoneatser').str.replace('^.*Crinum.*$', 'Crinum')\
    .str.replace('^.*Cystopteris.*$', 'Cystopteris').str.replace('^.*Daucus.*$', 'Daucus')\
    .str.replace('^.*Delph.*$', 'Delphinium').str.replace('^.*Dicksonia.*$', 'Dicksonia')\
    .str.replace('^.*Diervilla.*$', 'Diervilla').str.replace('^.*Dion.*$', 'Dionaea')\
    .str.replace('^.*Drosera.*$', 'Drosera').str.replace('^.*Acer palmatum.*$', 'Acer palmatum')\
    .str.replace('^.*Amelanchier.*$', 'Amelanchier').str.replace('^.*Amorpha.*$', 'Amorpha')\
    .str.replace('^.*Anisodontea.*$', 'Anisodontea').str.replace('^.*Annemona.*$', 'Annemona')\
    .str.replace('^.*Anthoxanthum.*$', 'Anthoxanthum').str.replace('^.*Apfel .*$', 'Apfel')\
    .str.replace('^.*Apfelquitte h.*$', 'Apfelquitte').str.replace('^.*Aronia .*$', 'Aronia')\
    .str.replace('^.*Buchs .*$', 'Buxus').str.replace('^.*Calamgrostis.*$', 'Calamagrostis')\
    .str.replace('^.*Chiastophyllum .*$', 'Chiastophyllum').str.replace('^.*Convolvulus .*$', 'Convolvulus')\
    .str.replace('^.*Ech .*$', 'Echinacea').str.replace('^.*Echinaea .*$', 'Echinacea')\
    .str.replace('^.*Echium.*$', 'Echium').str.replace('^.*Elodea .*$', 'Elodea')\
    .str.replace('^.*Epipactis.*$', 'Epipactis').str.replace('^.*Episetum .*$', 'Equisetum')\
    .str.replace('^.*Eragrostis.*$', 'Eragrostis').str.replace('^.*Eremus .*$', 'Eremurus')\
    .str.replace('^.*Erinus.*$', 'Erinus').str.replace('^.*Eriophorum .*$', 'Eriophorum')\
    .str.replace('^.*Eriophyllum.*$', 'Eriophyllum').str.replace('^.*Eucomis .*$', 'Eucomis')\
    .str.replace('^.*Euonymus.*$', 'Euonymus').str.replace('^.*Fontanalis .*$', 'Fontinalis')\
    .str.replace('^.*Forsythia.*$', 'Forsythia').str.replace('^.*Fruehlingsstauden .*$', 'Fruehlingsstauden')\
    .str.replace('^.*Fuchsien.*$', 'Fuchsia').str.replace('^.*Gailardia .*$', 'Gaillardia')\
    .str.replace('^.*Galium.*$', 'Galium').str.replace('^.*Gallium .*$', 'Galium')\
    .str.replace('^.*Galtonia.*$', 'Galtonia').str.replace('^.*Geranum .*$', 'Geranium')\
    .str.replace('^.*Gillenia.*$', 'Gillenia').str.replace('^.*Gladiolus .*$', 'Gladiolus')\
    .str.replace('^.*Globularia.*$', 'Globularia').str.replace('^.*Glyceria .*$', 'Glyceria')\
    .str.replace('^.*Glycyrrhiza.*$', 'Glycyrrhiza').str.replace('^.*Goniolimon .*$', 'Goniolimon')\
    .str.replace('^.*Gynostemma.*$', 'Gynostemma').str.replace('^.*Halimiocistus .*$', 'Halimiocistus')\
    .str.replace('^.*Hamamelis.*$', 'Hamamelis').str.replace('^.*Hellborus .*$', 'Helleborus')\
    .str.replace('^.*Helleboris.*$', 'Helleborus').str.replace('^.*Heuchea .*$', 'Heuchera')\
    .str.replace('^.*Horminum.*$', 'Horminum').str.replace('^.*Hysopus .*$', 'Hyssopus')\
    .str.replace('^.*IleG.*$', 'Ilex').str.replace('^.*Ilemes .*$', 'Ilex')\
    .str.replace('^.*Fontinalis.*$', 'Fontinalis').str.replace('^.*Indian Sunset.*$', 'Indian Sunset')\
    .str.replace('^.*Indigofera.*$', 'Indigofera').str.replace('^.*Inula.*$', 'Inula')\
    .str.replace('^.*Ipomoea.*$', 'Ipomoea').str.replace('^.*Isotoma.*$', 'Isotoma')\
    .str.replace('^.*Jasione.*$', 'Jasione').str.replace('^.*Jeffersonia.*$', 'Jeffersonia')\
    .str.replace('^.*Johannisbeere.*$', 'Johannisbeere').str.replace('^.*Juncus.*$', 'Juncus')\
    .str.replace('^.*Kirengeshoma.*$', 'Kirengeshoma').str.replace('^.*Kissenprimel.*$', 'Kissenprimel')\
    .str.replace('^.*Kletter-Erdbeere.*$', 'Kletter-Erdbeere').str.replace('^.*raeuter.*$', 'Kraeuter')\
    .str.replace('^.*Paeonie.*$', 'Paeonie').str.replace('^.*Funkie.*$', 'Funkie')\
    .str.replace('^.*habarber.*$', 'Rhabarber').str.replace('^.*Salbei.*$', 'Salbei')\
    .str.replace('^.*Traenendes Herz.*$', 'Traenendes Herz').str.replace('^.*elkenwurz.*$', 'Nelkenwurz')\
    .str.replace('^.*Laurus.*$', 'Laurus').str.replace('^.*Lavathera.*$', 'Lavatera')\
    .str.replace('^.*Lavendel.*$', 'Lavendel').str.replace('^.*Ledum.*$', 'Ledum')\
    .str.replace('^.*Leonorus.*$', 'Leonurus').str.replace('^.*Leonurus.*$', 'Leonurus')\
    .str.replace('^.*Leptodermis.*$', 'Leptodermis').str.replace('^.*Leucanth.*$', 'Leucanth')\
    .str.replace('^.*Leucanthemella.*$', 'Leucanthemella').str.replace('^.*Ligusticum.*$', 'Ligusticum')\
    .str.replace('^.*Lippia.*$', 'Lippia').str.replace('^.*Lobelia.*$', 'Lobelia')\
    .str.replace('^.*Lotus.*$', 'Lotos').str.replace('^.*Lycium.*$', 'Lycium')\
    .str.replace('^.*Marubium.*$', 'Marrubium').str.replace('^.*Mazus.*$', 'Mazus')\
    .str.replace('^.*Meconopsis.*$', 'Meconopsis').str.replace('^.*Melica.*$', 'Melica')\
    .str.replace('^.*Melissa.*$', 'Melissa').str.replace('^.*Melittis.*$', 'Melittis')\
    .str.replace('^.*Menyanthes.*$', 'Menyanthes').str.replace('^.*Meum.*$', 'Meum')\
    .str.replace('^.*Millium.*$', 'Millium').str.replace('^.*Molina.*$', 'Molinia')\
    .str.replace('^.*Montia.*$', 'Montia').str.replace('^.*Muehlenbeckia.*$', 'Muehlenbeckia')\
    .str.replace('^.*Mukdenia.*$', 'Mukdenia').str.replace('^.*Mukgenia.*$', 'Mukgenia')\
    .str.replace('^.*Musa.*$', 'Musa').str.replace('^.*Nasturtium.*$', 'Nasturtium')\
    .str.replace('^.*Nelumbo.*$', 'Nelumbo').str.replace('^.*Nympheae.*$', 'Nymphaea')\
    .str.replace('^.*Nymphoides.*$', 'Nymphoides').str.replace('^.*Ocimim.*$', 'Ocimum')\
    .str.replace('^.*Oenanthe.*$', 'Oenanthe').str.replace('^.*Onopordum.*$', 'Onopordum')\
    .str.replace('^.*Orchis.*$', 'Orchis').str.replace('^.*Origanum.*$', 'Origano')\
    .str.replace('^.*Penstomon.*$', 'Penstemon').str.replace('^.*Petasites.*$', 'Petasites')\
    .str.replace('^.*Petrorhagia.*$', 'Petrorhagia').str.replace('^.*Peucedanum.*$', 'Peucedanum')\
    .str.replace('^.*Phormium.*$', 'Phormium').str.replace('^.*Phragmites.*$', 'Phragmites')\
    .str.replace('^.*Phygelius.*$', 'Phygelius').str.replace('^.*Phyla.*$', 'Phyla')\
    .str.replace('^.*Phyllitis.*$', 'Phyllitis').str.replace('^.*Physalis.*$', 'Physalis')\
    .str.replace('^.*Physocarpus.*$', 'Physocarpus').str.replace('^.*Physostegia.*$', 'Physostegia')\
    .str.replace('^.*Phyteuma.*$', 'Phyteuma').str.replace('^.*Phytolacca.*$', 'Phytolacca')\
    .str.replace('^.*Pimpinella.*$', 'Pimpinella').str.replace('^.*Pink Paradise.*$', 'Pink Paradise')\
    .str.replace('^.*Pistia.*$', 'Pistia').str.replace('^.*Plat.*$', 'Platycodon')\
    .str.replace('^.*Pleioblastus.*$', 'Pleioblastus').str.replace('^.*Poa.*$', 'Poa')\
    .str.replace('^.*Podophyllum.*$', 'Podophyllum').str.replace('^.*Polygala.*$', 'Polygala')\
    .str.replace('^.*Polygonaum.*$', 'Polygonatum').str.replace('^.*Polypodium.*$', 'Polypodium')\
    .str.replace('^.*Preslia.*$', 'Preslia').str.replace('^.*Pritzelago.*$', 'Prizelago')\
    .str.replace('^.*Pseudolysimachion.*$', 'Pseudolysimachion').str.replace('^.*Pseudosasa.*$', 'Pseudosasa')\
    .str.replace('^.*Pteridium.*$', 'Pteridium').str.replace('^.*Pycanthemum.*$', 'Pycanthemum')\
    .str.replace('^.*Ramonda.*$', 'Ramonda').str.replace('^.*Ratibida.*$', 'Ratibida')\
    .str.replace('^.*Reynoutria.*$', 'Reynoutria').str.replace('^.*Rhabarber.*$', 'Rhabarber')\
    .str.replace('^.*Romneya.*$', 'Romneya').str.replace('^.*Roscoea.*$', 'Roscoea')\
    .str.replace('^.*Rosmarin.*$', 'Rosmarin').str.replace('^.*Rumeacetosa.*$', 'Rumeacetosa')\
    .str.replace('^.*Rumerugosus.*$', 'Rumerugosus').str.replace('^.*Rumesanguineus.*$', 'Rumesanguineus')\
    .str.replace('^.*Rungia.*$', 'Rungia').str.replace('^.*Ruta.*$', 'Ruta')\
    .str.replace('^.*Sagittaria.*$', 'Sagittaria').str.replace('^.*Salvinia.*$', 'Salvinia')\
    .str.replace('^.*Sanguinaria.*$', 'Sanguinaria').str.replace('^.*Sarracenia.*$', 'Sarracenia')\
    .str.replace('^.*Sasa.*$', 'Sasa').str.replace('^.*Satureja.*$', 'Satureja')\
    .str.replace('^.*Saururus.*$', 'Saururus').str.replace('^.*Schizachyrium.*$', 'Schizachyrium')\
    .str.replace('^.*Schoenoplectus.*$', 'Schoenoplectus').str.replace('^.*Scilla.*$', 'Scilla')\
    .str.replace('^.*Sedoro.*$', 'Sedoro').str.replace('^.*Selinum.*$', 'Selinum')\
    .str.replace('^.*Sempervivella.*$', 'Sempervivella').str.replace('^.*Sempervilla.*$', 'Sempervivella')\
    .str.replace('^.*Sempervivium.*$', 'Sempervivium').str.replace('^.*Sempervivvium.*$', 'Sempervivium')\
    .str.replace('^.*Senecio.*$', 'Senecio').str.replace('^.*Seseli.*$', 'Seseli')\
    .str.replace('^.*Silbersommer.*$', 'Silbersommer').str.replace('^.*Silphium.*$', 'Silphium')\
    .str.replace('^.*Fruehlingsstauden.*$', 'Fruehlingsstauden').str.replace('^.*Smilacina.*$', 'Smilacina')\
    .str.replace('^.*Solanum.*$', 'Solanum').str.replace('^.*Soldanella.*$', 'Soldanella')\
    .str.replace('^.*Solidaster.*$', 'Solidaster').str.replace('^.*Solidora.*$', 'Solidora')\
    .str.replace('^.*Solitaer.*$', 'Solitaer').str.replace('^.*Sommernachts.*$', 'Sommernachtstraum')\
    .str.replace('^.*Sorghastrum.*$', 'Sorghastrum').str.replace('^.*Sparganium.*$', 'Sparganium')\
    .str.replace('^.*Spartina.*$', 'Spartina').str.replace('^.*Spiraea.*$', 'Spiraea')\
    .str.replace('^.*Spirea.*$', 'Spiraea').str.replace('^.*Spodiopogon.*$', 'Spodiopogon')\
    .str.replace('^.*Sporobolus.*$', 'Sporobolus').str.replace('^.*Stachelbeere.*$', 'Stachelbeere')\
    .str.replace('^.*Staude .*$', 'Stauden').str.replace('^.*Steingartenstauden.*$', 'Steingartenstauden')\
    .str.replace('^.*Stellaria.*$', 'Stellaria').str.replace('^.*Stephanandra.*$', 'Stephanandra')\
    .str.replace('^.*Stevia.*$', 'Stevia').str.replace('^.*Stratoides.*$', 'Stratoides')\
    .str.replace('^.*Strobeliant.*$', 'Strobelianthes').str.replace('^.*Suesskirsche.*$', 'Suesskirsche')\
    .str.replace('^.*Sideritis.*$', 'Sideritis').str.replace('^.*Silybum.*$', 'Silybum')\
    .str.replace('^.*Syring.*$', 'Syringa').str.replace('^.*Silybum.*$', 'Silybum')\
    .str.replace('^.*Taxus.*$', 'Taxus').str.replace('^.*Thermopsis.*$', 'Thermopsis')\
    .str.replace('^.*Thymian.*$', 'Thymian').str.replace('^.*Trachystemon.*$', 'Trachystemon')\
    .str.replace('^.*Tropaeolum.*$', 'Tropaeolum').str.replace('^.*Uncinia.*$', 'Uncinia')\
    .str.replace('^.*Utricularia.*$', 'Utricularia').str.replace('^.*Veratrum.*$', 'Veratrum')\
    .str.replace('^.*Viburnum.*$', 'Viburnum').str.replace('^.*Yucca.*$', 'Yucca')\
    .str.replace('^.*Woodsia.*$', 'Woodsia').str.replace('^.*Zantedeschia.*$', 'Zantedeschia')\
    .str.replace('^.*Zitronen-Verbene.*$', 'Zitronenverbene').str.replace('^.*Zizania.*$', 'Zizania')\
    .str.replace('^.*Narzissus.*$', 'Narcissus').str.replace('^.*Crataegus.*$', 'Crataegus')\
    .str.replace('^.*Leptodernius.*$', 'Leptodermis').str.replace('^.*Ocimum.*$', 'Ocimum')\
    .str.replace('^.*Oregano.*$', 'Origano').str.replace('^.*Phuopsis.*$', 'Phuopsis')\
    .str.replace('^.*Rose.*$', 'Rose').str.replace('^.*Carpinus.*$', 'Carpinus')\
    .str.replace('^.*Crataegus.*$', 'Crataegus').str.replace('^.*Cytisus.*$', 'Cytisus')\
    .str.replace('^.*Himbeeren.*$', 'Himbeeren').str.replace('^.*Juglans.*$', 'Juglans')\
    .str.replace('^.*Wallnuss.*$', 'Juglans').str.replace('^.*Weigela.*$', 'Weigela')\
    .str.replace('^.*aster.*$', 'Aster').str.replace('^.*avandula .*$', 'Lavandula')\
    .str.replace('^.*Lavendel.*$', 'Lavandula').str.replace('^.*helleborus .*$', 'Helleborus')\
    .str.replace('^.*hosta l blau.*$', 'Funkie').str.replace('^.*arcissus .*$', 'Narcissus')\
    .str.replace('^.*alchemilla.*$', 'Alchemilla').str.replace('^.*Terracotta .*$', 'Achillea')\
    .str.replace('^.*Sempervivvum.*$', 'Sempervivum').str.replace('^.*Sempervivium .*$', 'Sempervivum')\
    .str.replace('^.*Robinia.*$', 'Robinia').str.replace('^.*Salicaprea .*$', 'Salicaprea')\
    .str.replace('^.*Philadelphus.*$', 'Philadelphus').str.replace('^.*Micromeria .*$', 'Micromeria')\
    .str.replace('^.*Pagels kl Elfenblume.*$', 'Epimedium').str.replace('^.*Ligustrum .*$', 'Ligustrum')

file_df = file_df.loc[:,~(file_df.columns.str.contains('^.*Fracht.*$', case=False)
                          | file_df.columns.str.contains('^.*Angebotspaket.*$', case=False)
                          | file_df.columns.str.contains('^.*Werbe.*$', case=False)
                          | file_df.columns.str.contains('^.*Vita.Verde.*$', case=False)
                          | file_df.columns.str.contains('^.*zahlung.*$', case=False)
                          | file_df.columns.str.contains('^.*Europalette.*$', case=False)
                          | file_df.columns.str.contains('^.*gebotspaket.*$', case=False)
                          | file_df.columns.str.contains('^.*Aufkleber fuer Saeule.*$', case=False)
                          | file_df.columns.str.contains('^.*Aufsetzer.*$', case=False)
                          | file_df.columns.str.contains('^.*Ausstellen der Pflanzen in die Beete  pauschal.*$', case=False)
                          | file_df.columns.str.contains('^.*Ausstellungsflaeche.*$', case=False)
                          | file_df.columns.str.contains('^.*Auswaschen.*$', case=False)
                          | file_df.columns.str.contains('^.*Bild.*$', case=False)
                          | file_df.columns.str.contains('^.*tikett.*$', case=False)
                          | file_df.columns.str.contains('^.*Commission  % \(=Gebuehr fuereuheitenpromotion i.*$', case=False)
                          | file_df.columns.str.contains('^.*Doppelstecker Schwimmpflanzen.*$', case=False)
                          | file_df.columns.str.contains('^.*Dstauden iS  ccm xFragarua vv | xlliu.*$', case=False)
                          | file_df.columns.str.contains('^.*Aktueller Spar-Mix-CC.*$', case=False)
                          | file_df.columns.str.contains('^.*Boegen Klebeecken.*$', case=False)
                          | file_df.columns.str.contains('^.*Einwegtrays geliefert.*$', case=False)
                          | file_df.columns.str.contains('^.*Gartenschaetze.*$', case=False)
                          | file_df.columns.str.contains('^.*Gittertopf.*$', case=False)
                          | file_df.columns.str.contains('^.*Gutschrift.*$', case=False)
                          | file_df.columns.str.contains('^.*Handzettel.*$', case=False)
                          | file_df.columns.str.contains('^.*Karton.*$', case=False)
                          | file_df.columns.str.contains('^.*Klebepunkte.*$', case=False)
                          | file_df.columns.str.contains('^.*kisten.*$', case=False)
                          | file_df.columns.str.contains('^.*Musterpflanzen.*$', case=False)
                          | file_df.columns.str.contains('^.*Palettenufsaetze.*$', case=False)
                          | file_df.columns.str.contains('^.*Pauschale fuer dasusstellen.*$', case=False)
                          | file_df.columns.str.contains('^.*Pflanzen.*$', case=False)
                          | file_df.columns.str.contains('^.*Porto.*$', case=False)
                          | file_df.columns.str.contains('^.*Praesentationstafel.*$', case=False)
                          | file_df.columns.str.contains('^.*Werbung.*$', case=False)
                          | file_df.columns.str.contains('^.*Prospekt.*$', case=False)
                          | file_df.columns.str.contains('^.*Sonderrabatt.*$', case=False)
                          | file_df.columns.str.contains('^.*Sonstige.*$', case=False)
                          | file_df.columns.str.contains('^.*kosten.*$', case=False)
                          | file_df.columns.str.contains('^.*Kiste.*$', case=False)
                          | file_df.columns.str.contains('^.*Bonus.*$', case=False)
                          | file_df.columns.str.contains('^.*SPAR-MIX-CC.*$', case=False)
                          | file_df.columns.str.contains('^.*Unnamed.*$', case=False)
                          | file_df.columns.str.contains('^.*sed pur.*$', case=False)
                          | file_df.columns.str.contains('^.*hohestern.*$', case=False)
                          | file_df.columns.str.contains('^.*Rhododendron-Schaden.*$', case=False)
                          | file_df.columns.str.contains('^.*Rhododendron-Schaden.*$', case=False)
                          )]

file_df = file_df.groupby(file_df.columns, axis=1).sum()
file_df.index = pd.to_datetime(file_df.index)

idx = pd.date_range('2001-01-11', '28.10.2020')

file_df = file_df.reindex(idx)
file_df.index.name = "Auf. Datum"
print(file_df.columns)
file_df.to_csv("Schachtschneider_further.csv")