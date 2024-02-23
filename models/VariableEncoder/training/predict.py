import tqdm
import torch
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from training.lightning import TrainModule

class EncoderDataset(Dataset):
        def __init__(self, data_array: torch.Tensor):
            self.data_array = data_array
        def __len__(self):
            return len(self.data_array)
        def __getitem__(self, idx):
            sample = self.data_array[idx]
            return sample

class VariablePredictor:
    def __init__(self, model_path: str, batch_size: int):
        self.model_path = model_path
        self.batch_size = batch_size
        self.load_models(model_path)
        device = ("cuda" if torch.cuda.is_available() else "cpu" )
        self.device = torch.device(device)
    
    def load_models(self, model_path):
        folder_path = Path(model_path)
        first_file = next(folder_path.iterdir(), None)
        if first_file:
                self.model = TrainModule.load_from_checkpoint(first_file).model
                self.model.init_seq(self.device)
        else:
            raise Exception("Not exists VariableEncoder model")
    
    def predict(self, dataset: torch.Tensor) -> torch.Tensor:
        dataset = EncoderDataset(dataset)
        data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=8)
        predictions = []
        for _, batch in enumerate(tqdm.tqdm(data_loader)):
            # shape = (batch, var_len, hidden)
            predict = self.model.encode(batch.to(self.device)).cpu()
            predictions.append(predict)
        
        # predictions.shape = (time, var_len, hidden)
        predictions = torch.cat(predictions, dim=0)
        return predictions

