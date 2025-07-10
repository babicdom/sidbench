import argparse

from models import get_model
from models.models import MODELS

from dataset.dataset_paths import DATASET_PATHS

from evaluate import run_for_model

from options import EvalOptions
from utils.util import set_random_seed

SEED = 0

JPEG_QUALITY = [95, 75, 50, 25]
GAUSSIAN_SIGMA = [2, 4]


if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser = EvalOptions().initialize(parser)

    opt = parser.parse_args()

    if opt.family == "all":
        datasets = [
            dict(
                data_paths=['data/test'],
                source='all',
                family='all',
                generative_model='all'
            ),
        ]
    elif opt.family == "gan":
        datasets = [
            dict(
                data_paths=[
                    'data/test/diffusion_datasets/laion',
                    'data/test/biggan/',
                    'data/test/cyclegan',
                    'data/test/gaugan',
                    'data/test/progan',
                    'data/test/stargan',
                    'data/test/stylegan',
                    'data/test/stylegan2',
                    'data/test/deepfake',
                    'data/test/crn',
                    'data/test/imle',
                    'data/test/san',
                    'data/test/seeingdark',
                    'data/test/whichfaceisreal',
                    'data/test/diffusion_datasets/dalle',
                    'data/test/synthbuster/dalle3/',
                    'data/test/synthbuster/dalle2/',
                    'data/test/spai/gigagan/',
                ],
                source='all gans',
                family='gan based',
                generative_model='gan based'
            )
        ]
    elif opt.family == 'ldm':
        datasets = [
            dict(
                data_paths=[
                    'data/test/diffusion_datasets/laion',
                    'data/test/synthbuster/raise',
                    'data/test/diffusion_datasets/imagenet',
                    'data/test/diffusion_datasets/glide_100_10',
                    'data/test/diffusion_datasets/glide_100_27',
                    'data/test/diffusion_datasets/glide_50_27',
                    'data/test/diffusion_datasets/guided',
                    'data/test/diffusion_datasets/ldm_100',
                    'data/test/diffusion_datasets/ldm_200',
                    'data/test/diffusion_datasets/ldm_200_cfg',
                    'data/test/synthbuster/glide/',
                    'data/test/synthbuster/dalle2/',
                    'data/test/synthbuster/stable-diffusion-1-3/',
                    'data/test/synthbuster/stable-diffusion-1-4/',
                    'data/test/synthbuster/midjourney-v5/',
                    'data/test/synthbuster/dalle3/',
                    'data/test/synthbuster/stable-diffusion-2/',
                    'data/test/synthbuster/stable-diffusion-xl/',
                    'data/test/synthbuster/firefly/',
                    'data/test/spai/flux/',
                    'data/test/spai/midjourney-v6.1/',
                    'data/test/spai/stable-diffusion-3/', 
                ],
                source='all ldms',
                family='ldm based',
                generative_model='ldm based'
            )
        ]
    else:
        datasets = [
            dict(data_paths=[dp['real_path'], dp['fake_path']], 
                source=dp['source'],
                generative_model=dp['generative_model'],
                family=dp['family'])
            for dp in DATASET_PATHS
        ]
    print(f"Number of datasets: {len(datasets)}")
    print(f"Options: {opt}")

    for model_params in MODELS:
        set_random_seed()
        print('Model: ', model_params['modelName'] if 'desc' not in model_params else model_params['modelName'] + model_params['desc'])

        opt.modelName = model_params['modelName']
        opt.ckpt = model_params['ckpt']
        opt.experiment = model_params.get('experiment', None)
        opt.desc = model_params.get('desc', '')
        model = get_model(opt)

        print(f'Cropping: {opt.cropSize}, Image Size: {opt.imgSize}')
        print('\tjpeg_quality: ', None, 'gaussian_sigma: ', None)
        opt.gaussianSigma = None
        opt.jpegQuality = None
        run_for_model(datasets=datasets, model=model, opt=opt)

        if opt.testJPEGQuality:
            for jpeg_quality in JPEG_QUALITY:
                print('\tjpeg_quality: ', jpeg_quality)
                opt.gaussianSigma = None
                opt.jpegQuality = jpeg_quality
                run_for_model(datasets=datasets, model=model, opt=opt)
        
        if opt.testGaussianSigma:
            for gaussian_sigma in GAUSSIAN_SIGMA:
                print('\tgaussian_sigma: ', gaussian_sigma)
                opt.gaussianSigma = gaussian_sigma
                opt.jpegQuality = None
                run_for_model(datasets=datasets, model=model, opt=opt)
        
        # print('\tjpeg_quality: ', 50, 'gaussian_sigma: ', 2)
        # opt.gaussianSigma = 2
        # opt.jpegQuality = 50
        # run_for_model(datasets=datasets, model=model, opt=opt)