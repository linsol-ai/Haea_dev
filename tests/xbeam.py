import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
import xarray_beam as xbeam
import xarray as xr
import zarr
import fsspec
class FilterAndProcessData(beam.DoFn):
    def process(self, element, time_range, lat_range, lon_range):
        # Zarr 데이터셋을 열기
        ds = xr.open_zarr(fsspec.get_mapper('gcs://' + element), consolidated=True)
        
        # 시간, 위도, 경도 범위에 따른 필터링
        ds_filtered = ds.sel(time=slice(*time_range), lat=slice(*lat_range), lon=slice(*lon_range))
        
        # 필요한 전처리 로직 추가
        # 예: ds_processed = ds_filtered.mean(dim='time')
        
        yield ds_filtered

def run():
    # 파이프라인 옵션 설정
    options = PipelineOptions(
        runner='DataflowRunner',
        project='YOUR_PROJECT_ID',
        temp_location='gs://YOUR_BUCKET_NAME/temp',
        region='YOUR_REGION'
    )
    
    # 파이프라인 정의
    with beam.Pipeline(options=options) as p:
        # GCS에서 Zarr 파일 목록을 읽음
        zarr_files = ['gs://YOUR_BUCKET_NAME/path/to/your/data.zarr']
        
        # Zarr 데이터 처리
        (p | 'CreateFileList' >> beam.Create(zarr_files)
           | 'FilterAndProcess' >> beam.ParDo(FilterAndProcessData(), time_range=('2023-01-01', '2023-01-31'), 
                                               lat_range=(30, 50), lon_range=(-130, -60))
           # 전처리된 데이터를 출력하거나 저장할 액션 추가
           # 예: | 'WriteResults' >> beam.io.WriteToText('gs://YOUR_BUCKET_NAME/output/')
        )

if __name__ == '__main__':
    run()