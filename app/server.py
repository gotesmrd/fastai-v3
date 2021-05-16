import aiohttp
import asyncio
import uvicorn
from fastai import *
from fastai.vision import *
from io import BytesIO
from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import HTMLResponse, JSONResponse
from starlette.staticfiles import StaticFiles

export_file_url = 'https://www.dropbox.com/s/vrhs4q9i98a6bs1/model.pkl?dl=0'
export_file_name = 'model.pkl'

classes = ['Affenpinscher',
 'Afghan_hound',
 'African_hunting_dog',
 'Airedale',
 'American_staffordshire_terrier',
 'Appenzeller',
 'Australian_shepherd',
 'Australian_terrier',
 'Basenji',
 'Basset',
 'Beagle',
 'Bedlington_terrier',
 'Bernese_mountain_dog',
 'Bichon_frise',
 'Black_and_tan_coonhound',
 'Black_sable',
 'Blenheim_spaniel',
 'Bloodhound',
 'Bluetick',
 'Border_collie',
 'Border_terrier',
 'Borzoi',
 'Boston_bull',
 'Bouvier_des_flandres',
 'Boxer',
 'Brabancon_griffo',
 'Briard',
 'Brittany_spaniel',
 'Bull_mastiff',
 'Cairn',
 'Cane_carso',
 'Cardigan',
 'Chesapeake_bay_retriever',
 'Chihuahua',
 'Chinese_crested_dog',
 'Chinese_rural_dog',
 'Chow',
 'Clumber',
 'Cocker_spaniel',
 'Collie',
 'Curly_coated_retriever',
 'Dandie_dinmont',
 'Dhole',
 'Dingo',
 'Doberman',
 'English_foxhound',
 'English_setter',
 'English_springer',
 'Entlebucher',
 'Eskimo_dog',
 'Fila braziliero',
 'Flat_coated_retriever',
 'French_bulldog',
 'German_shepherd',
 'German_short_haired_pointer',
 'Giant_schnauzer',
 'Golden_retriever',
 'Gordon_setter',
 'Great_dane',
 'Great_pyrenees',
 'Greater_swiss_mountain_dog',
 'Groenendael',
 'Ibizan_hound',
 'Irish_setter',
 'Irish_terrier',
 'Irish_water_spaniel',
 'Irish_wolfhound',
 'Italian_greyhound',
 'Japanese_spaniel',
 'Japanese_spitzes',
 'Keeshond',
 'Kelpie',
 'Kerry_blue_terrier',
 'Komondor',
 'Kuvasz',
 'Labrador_retriever',
 'Lakeland_terrier',
 'Leonberg',
 'Lhasa',
 'Malamute',
 'Malinois',
 'Maltese_dog',
 'Mexican_hairless',
 'Miniature_pinscher',
 'Miniature_poodle',
 'Miniature_schnauzer',
 'Newfoundland',
 'Norfolk_terrier',
 'Norwegian_elkhound',
 'Norwich_terrier',
 'Old_english_sheepdog',
 'Otterhound',
 'Papillon',
 'Pekinese',
 'Pembroke',
 'Pomeranian',
 'Pug',
 'Redbone',
 'Rhodesian_ridgeback',
 'Rottweiler',
 'Saint_bernard',
 'Saluki',
 'Samoyed',
 'Schipperke',
 'Scotch_terrier',
 'Scottish_deerhound',
 'Sealyham_terrier',
 'Shetland_sheepdog',
 'Shiba_dog',
 'Shih_tzu',
 'Siberian_husky',
 'Silky_terrier',
 'Soft_coated_wheaten_terrier',
 'Staffordshire_bullterrier',
 'Standard_poodle',
 'Standard_schnauzer',
 'Sussex_spaniel',
 'Tibetan_mastiff',
 'Tibetan_terrier',
 'Toy_poodle',
 'Toy_terrier',
 'Vizsla',
 'Walker_hound',
 'Weimaraner',
 'Welsh_springer_spaniel',
 'West_highland_white_terrier',
 'Whippet',
 'Wire_haired_fox_terrier',
 'Yorkshire_terrier']
path = Path(__file__).parent

app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory='app/static'))


async def download_file(url, dest):
    if dest.exists(): return
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.read()
            with open(dest, 'wb') as f:
                f.write(data)


async def setup_learner():
    await download_file(export_file_url, path / export_file_name)
    try:
        learn = load_learner(path, export_file_name)
        return learn
    except RuntimeError as e:
        if len(e.args) > 0 and 'CPU-only machine' in e.args[0]:
            print(e)
            message = "\n\nThis model was trained with an old version of fastai and will not work in a CPU environment.\n\nPlease update the fastai library in your training environment and export your model again.\n\nSee instructions for 'Returning to work' at https://course.fast.ai."
            raise RuntimeError(message)
        else:
            raise


loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_learner())]
learn = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()


@app.route('/')
async def homepage(request):
    html_file = path / 'view' / 'index.html'
    return HTMLResponse(html_file.open().read())


@app.route('/analyze', methods=['POST'])
async def analyze(request):
    img_data = await request.form()
    img_bytes = await (img_data['file'].read())
    img = open_image(BytesIO(img_bytes))
    prediction = learn.predict(img)[0]
    return JSONResponse({'result': str(prediction)})


if __name__ == '__main__':
    if 'serve' in sys.argv:
        uvicorn.run(app=app, host='0.0.0.0', port=5000, log_level="info")
