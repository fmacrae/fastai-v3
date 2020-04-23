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

export_file_url = 'https://websofttechnology.s3.amazonaws.com/MLCLub/food.pkl'
export_file_name = 'food.pkl'

classes = [ 'rice ', 	 'eels on rice ', 	 'pilaf ', 	 'chicken- n -egg on rice ', 	 'pork cutlet on rice ', 	 'beef curry ', 	 'sushi ', 	 'chicken rice ', 	 'fried rice ', 	 'tempura bowl ', 	 'bibimbap ', 	 'toast ', 	 'croissant ', 	 'roll bread ', 	 'raisin bread ', 	 'chip butty ', 	 'hamburger ', 	 'pizza ', 	 'sandwiches ', 	 'udon noodle ', 	 'tempura udon ', 	 'soba noodle ', 	 'ramen noodle ', 	 'beef noodle ', 	 'tensin noodle ', 	 'fried noodle ', 	 'spaghetti ', 	 'Japanese-style pancake ', 	 'takoyaki ', 	 'gratin ', 	 'sauteed vegetables ', 	 'croquette ', 	 'grilled eggplant ', 	 'sauteed spinach ', 	 'vegetable tempura ', 	 'miso soup ', 	 'potage ', 	 'sausage ', 	 'oden ', 	 'omelet ', 	 'ganmodoki ', 	 'jiaozi ', 	 'stew ', 	 'teriyaki grilled fish ', 	 'fried fish ', 	 'grilled salmon ', 	 'salmon meuniere  ', 	 'sashimi ', 	 'grilled pacific saury  ', 	 'sukiyaki ', 	 'sweet and sour pork ', 	 'lightly roasted fish ', 	 'steamed egg hotchpotch ', 	 'tempura ', 	 'fried chicken ', 	 'sirloin cutlet  ', 	 'nanbanzuke ', 	 'boiled fish ', 	 'seasoned beef with potatoes ', 	 'hambarg steak ', 	 'beef steak ', 	 'dried fish ', 	 'ginger pork saute ', 	 'spicy chili-flavored tofu ', 	 'yakitori ', 	 'cabbage roll ', 	 'rolled omelet ', 	 'egg sunny-side up ', 	 'fermented soybeans ', 	 'cold tofu ', 	 'egg roll ', 	 'chilled noodle ', 	 'stir-fried beef and peppers ', 	 'simmered pork ', 	 'boiled chicken and vegetables ', 	 'sashimi bowl ', 	 'sushi bowl ', 	 'fish-shaped pancake with bean jam ', 	 'shrimp with chill source ', 	 'roast chicken ', 	 'steamed meat dumpling ', 	 'omelet with fried rice ', 	 'cutlet curry ', 	 'spaghetti meat sauce ', 	 'fried shrimp ', 	 'potato salad ', 	 'green salad ', 	 'macaroni salad ', 	 'Japanese tofu and vegetable chowder ', 	 'pork miso soup ', 	 'chinese soup ', 	 'beef bowl ', 	 'kinpira-style sauteed burdock ', 	 'rice ball ', 	 'pizza toast ', 	 'dipping noodles ', 	 'hot dog ', 	 'french fries ', 	 'mixed rice ', 	 'goya chanpuru ']


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
    prediction, *rest = learn.predict(img)
    
    
    prediction_str = classes[int(prediction.obj)-1]
    return JSONResponse({'result': str(prediction_str)+ ' ' + str(rest) })


if __name__ == '__main__':
    if 'serve' in sys.argv:
        uvicorn.run(app=app, host='0.0.0.0', port=5000, log_level="info")
