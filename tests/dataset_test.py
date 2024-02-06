import xarray_beam as xbeam
import xarray
from apache_beam.options.pipeline_options import PipelineOptions
import apache_beam as beam

# Apache Beam 파이프라인 옵션 설정
pipeline_options = PipelineOptions(
    runner='DirectRunner',  # 로컬에서 실행하는 경우 'DirectRunner' 사용
    project='YOUR_PROJECT_ID',  # GCP 프로젝트 ID 설정
    region='YOUR_REGION',  # GCP 리전 설정
    temp_location='gs://YOUR_BUCKET/temp',  # 임시 파일 저장 위치 설정
)

# Xarray-Beam 파이프라인 정의
def run_pipeline():
    with beam.Pipeline(options=pipeline_options) as p:
        _ = (
            p
            | 'Read NetCDF' >> xbeam.open_zarr('gs://weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr')  # NetCDF 파일 읽기
            | 'Dataset to chunks' >> xbeam.DatasetToChunks()  # 데이터셋을 청크로 분할
            | 'Calculate mean' >> beam.Map(lambda chunk: chunk.mean(dim='time'))  # 시간에 따른 평균 계산
            | 'Chunks to dataset' >> xbeam.ChunksToDataset()  # 결과 청크를 데이터셋으로 병합
            | 'Write results' >> xbeam.WriteNetcdf('path/to/output/data.nc')  # 결과 NetCDF 파일로 저장
        )


# 파이프라인 실행
if __name__ == '__main__':
    run_pipeline()