import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
import xarray
import xarray_beam

# GCS 경로 설정
GCS_BUCKET = 'dataflow_preprocess'
INPUT_ZARR_PATH = 'gs://weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721.zarr'
OUTPUT_ZARR_PATH = f'gs://{GCS_BUCKET}/result/1440x721.zarr'

# 파이프라인 옵션 설정
options = PipelineOptions(
    runner='DirectRunner',
)

class ReadZarrData(beam.DoFn):
    def process(self, element, latitude, longitude, start_time, end_time):
        # Zarr 데이터셋 열기
        ds = xarray.open_zarr(store)

        # 특정 위경도 및 시간으로 데이터 필터링
        ds_filtered = ds.sel(lat=latitude, lon=longitude, time=slice(start_time, end_time))

        # 필터링된 데이터를 메모리에 저장
        filtered_data = ds_filtered.compute()

        # 필터링된 데이터를 JSON 또는 기타 형식으로 변환하여 반환
        # 예제에서는 단순화를 위해 데이터의 요약 정보만 반환합니다.
        yield filtered_data.to_dict()



def preprocess_dataset(key: xarray_beam.Key, dataset: xarray.Dataset):
    ds = dataset
    ds_filtered = ds.sel(time=slice('2023-01-01', '2023-01-31'), latitude=slice(32.2, 39.0), longitude=slice(124.2, 131))
    return key, ds_filtered

def run():
    source_dataset, source_chunks = xarray_beam.open_zarr(INPUT_ZARR_PATH)
    template = (
      xarray_beam.make_template(source_dataset)
      .sel(time=slice('2023-01-01', '2023-01-31'), latitude=slice(32.2, 39.0), longitude=slice(124.2, 131))
   ) 
    with beam.Pipeline() as p:
        _ = (
            p
            | 'ChunkingDataset' >> xarray_beam.DatasetToChunks(source_dataset, chunks=source_chunks)
            | xarray_beam.SplitChunks({'time': 1})
            | 'PreprocessDataset' >> beam.MapTuple(preprocess_dataset)
            | 'WriteZarrToGCS' >> xarray_beam.ChunksToZarr('/workspace/Haea/tests/1440x721.zarr', template, source_chunks)
        )

        

if __name__ == '__main__':
    run()
