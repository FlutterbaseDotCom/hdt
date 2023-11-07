import os


def get_pretrained_model(model, pretrained_model_name):
    model_dir = './downloaded_models'
    model_path = os.path.join(model_dir, f'{pretrained_model_name}.bin')
    
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)
    
    if not os.path.isfile(model_path):
        model_url = f"https://storage.googleapis.com/yakiv-dt-public/models/{pretrained_model_name}.bin"
        download(model_url, model_path)
    
    # Load the model from the path
    loaded_model = model.load(model_path)
    
    return loaded_model