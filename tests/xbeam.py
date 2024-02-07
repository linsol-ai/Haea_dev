import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
import xarray_beam as xbeam
import xarray as xr
import zarr
import gcsfs

class PreprocessERA5Data(beam.DoFn):
    def process(self, element, bucket_name, variable, latitude_range, longitude_range, time_range):
        fs = gcsfs.GCSFileSystem(project='genfit-7ba0d')
        store = fs.get_mapper(f'gs://{bucket_name}/{element}')
        ds = xr.open_zarr(store)
        
        # 경위도 및 시간대에 따라 데이터 필터링
        ds_filtered = ds.sel(latitude=slice(*latitude_range), longitude=slice(*longitude_range), time=slice(*time_range))
        
        # 선택된 변수만 유지
        ds_filtered = ds_filtered[[variable]]
        
        # 결과를 Zarr 파일로 저장
        output_store = fs.get_mapper(f'gs://{bucket_name}/preprocessed/{element}_preprocessed.zarr')
        ds_filtered.to_zarr(output_store, mode='w')
        
        yield f'Processed and saved: {element}_preprocessed.zarr'

def run_pipeline(bucket_name, input_filename, variable, latitude_range, longitude_range, time_range):
    pipeline_options = PipelineOptions(
        runner='DataflowRunner',
        project='genfit-7ba0d',
        temp_location=f'gs://{bucket_name}/temp',
        region='us-east1',
        requirements_file=./requirements.txt
    )
    
    with beam.Pipeline(options=pipeline_options) as p:
        (
            p
            | 'CreateFileList' >> beam.Create([input_filename])
            | 'PreprocessData' >> beam.ParDo(PreprocessERA5Data(), bucket_name, variable, latitude_range, longitude_range, time_range)
        )

# 파이프라인 실행
if __name__ == '__main__':
    bucket_name = 'dataflow_preprocess'
    input_filename = 'gs://weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721.zarr'  # Zarr 파일 경로
    variable = 'geopotential'
    latitude_range = (-10, 10)  # 예시 경위도 범위
    longitude_range = (100, 120)  # 예시 경도 범위
    time_range = ('2020-01-01', '2020-12-31')  # 예시 시간 범위
    
    run_pipeline(bucket_name, input_filename, variable, latitude_range, longitude_range, time_range)