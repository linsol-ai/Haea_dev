from pathlib import Path

class VariablePredictor:
    def __init__(self, model_path: str, batch_size: int):
        self.model_path = model_path
        self.batch_size
    
    def load_models(self, variables, model_path):
        models = {}
        for key in variables:
            folder_path = Path(os.path.join(model_path, key))
            first_file = next(folder_path.iterdir(), None)
            if first_file:
                print(f"====== LOAD MODELS : {key} =======")
                model = DVAETrainModule.load_from_checkpoint(first_file)
                models[key] = model
            else:
                print("변수 폴더가 비어있습니다.")
                break

        return models