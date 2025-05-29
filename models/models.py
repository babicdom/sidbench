
MODELS = [
    # {
    #     "modelName": "UnivFD",
    #     "trainedOn": "progan",
    #     "ckpt": "./weights/univfd/fc_weights.pth"
    # },
    # {
    #     "modelName": "CNNDetect",
    #     "trainedOn": "progan",
    #     "ckpt": "./weights/cnndetect/blur_jpg_prob0.5.pth"
    # },
    # {
    #     "modelName": "DIMD",
    #     "trainedOn": "latent_diffusion",
    #     "ckpt": "./weights/dimd/corvi22_latent_model.pth",
    #     "desc": "latent_diffusion"
    # },
    # {
    #     "modelName": "DIMD",
    #     "trainedOn": "progan",
    #     "ckpt": "./weights/dimd/corvi22_progan_model.pth",
    #     "desc": "progan"
    # },
    # {
    #     "modelName": "DIMD",
    #     "trainedOn": "stylegan2",
    #     "ckpt": "./weights/dimd/gandetection_resnet50nodown_stylegan2.pth",
    #     "desc": "stylegan2"
    # },
    # {
    #     "modelName": "DeFake",
    #     "trainedOn": "SD",
    #     "ckpt": "./weights/defake/clip_linear.pth",
    # },
    # {
    #     "modelName": "LGrad",
    #     "trainedOn": "progan",
    #     "ckpt": "./weights/lgrad/LGrad.pth"
    # },
    # {
    #     "modelName": "FreqDetect",
    #     "trainedOn": "progan",
    #     "ckpt": "./weights/freqdetect/DCTAnalysis.pth"
    # },
    # {
    #     "modelName": "Rine",
    #     "trainedOn": "progan",
    #     "ckpt": "./weights/rine_original/model_1class_trainable.pth",
    #     "ncls": "1class",
    #     "desc": "_1_class"
    # },
    # {
    #     "modelName": "Rine",
    #     "trainedOn": "progan",
    #     "ckpt": "./weights/rine_original/model_2class_trainable.pth",
    #     "ncls": "2class",
    #     "desc": "_2_class"
    # },
    # {
    #     "modelName": "Rine",
    #     "trainedOn": "progan",
    #     "ckpt": "./weights/rine_original/model_4class_trainable.pth",
    #     "ncls": "4class",
    #     "desc": "_4_class"
    # },
    # {
    #     "modelName": "Rine",
    #     "trainedOn": "latent_diffusion",
    #     "ckpt": "./weights/rine_original/model_ldm_trainable.pth",
    #     "ncls": "ldm",
    #     "desc": "_latent_diffusion"
    # },
    # {
    #     "modelName": "NPR",
    #     "trainedOn": "progan",
    #     "ckpt": "./weights/npr/NPR.pth"
    # },
    # {
    #     "modelName": "Fusing",
    #     "trainedOn": "progan",
    #     "ckpt": "./weights/fusing/PSM.pth"
    # },
    # {
    #     "modelName": "GramNet",
    #     "trainedOn": "progan",
    #     "ckpt": "./weights/gramnet/Gram.pth"
    # },
    # {
    #     "modelName": "Dire",
    #     "trainedOn": "progan",
    #     "ckpt": "./weights/gramnet/Gram.pth"
    # },
    # {
    #     "modelName": "SPAI",
    #     "trainedOn": "progan",
    #     "ckpt": "./weights/spai/spai.pth"
    # },
    # {
    #     "modelName": "CLIPformer",
    #     "trainedOn": "progan",
    #     "ckpt": "./weights/CLIPformer/train.pth"
    # },
    # {
    #     "modelName": "IntermediatePatch",
    #     "trainedOn": "progan",
    #     "ckpt": "./weights/IntermediatePatch/train_progan.pth",
    #     "experiment": "./weights/IntermediatePatch/experiment_progan.pickle",
    # },
    # {
    #     "modelName": "IntermediatePatch",
    #     "trainedOn": "ldm",
    #     "ckpt": "./weights/IntermediatePatch/train_progan_supcon.pth",
    #     "experiment": "./weights/IntermediatePatch/experiment_progan_supcon.pickle",
    #     "desc": "Progan_max_supcon",
    # },
    # {
    #     "modelName": "AttentionIntermediatePatch",
    #     "trainedOn": "progan",
    #     "ckpt": "./weights/IntermediatePatch/train_attention_supcon.pth",
    #     "experiment": "./weights/IntermediatePatch/experiment_attention_supcon.pickle",
    #     "desc": "_progan_supcon",
    # },
    {
        "modelName": "SigLIPIntermediate",
        "trainedOn": "progan",
        "ckpt": "./weights/IntermediatePatch/train_siglip.pth",
        "experiment": "./weights/IntermediatePatch/experiment_siglip.json",
    },
]

