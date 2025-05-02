# from airflow import DAG
# from airflow.decorators import task
from datetime import datetime
# from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator
from pyspark import SparkConf
from pyspark.sql import SparkSession

default_args = {
    'owner': 'gis_team',
    'depends_on_past': False,
    'start_date': datetime(2023, 1, 1),
    'retries': 1
}

# @task
# def check_new_shapefiles(bucket: str):
#     # Your file checking logic here
#     return None
#
# @task
# def validate_results():
#     # Your validation logic here
#     pass

# with DAG('shapefile_processing',
#          default_args=default_args,
#          schedule='@daily',
#          catchup=False) as dag:
#     # Task 1: Check for new shapefiles
#     check_files = task(
#         task_id='check_new_shapefiles',
#         # python_callable=check_new_shapefiles,
#         op_kwargs={'bucket': 'gis-data-bucket'}
#     )
#
#     # Task 2: Process with Spark
#     process_shapefile = SparkSubmitOperator(
#         task_id='process_shapefile_spark',
#         application='/path/to/spark_shapefile_processor.py',
#         application_args=['{{ ti.xcom_pull(task_ids="check_new_shapefiles") }}'],
#         conn_id='spark_default',
#         executor_memory='4G',
#         driver_memory='2G'
#     )
#
#     # Task 3: Validate output
#     validate_output = task(
#         task_id='validate_processed_data',
#         # python_callable=validate_results
#     )
#
#     check_files >> process_shapefile >> validate_output
#
#     # spark_shapefile_processor.py





def main(shapefile_path):
    # Configure Spark with GeoTools support
    conf = SparkConf().setAppName("shapefile-loader").setMaster("local")
    # conf = geopyspark_conf(master="local[*]", appName="shapefile-loader")
    spark = SparkSession.builder.config(conf=conf).getOrCreate()

    # Using GeoTools for shapefile reading
    df = spark.read \
        .format("geotools") \
        .option("type", "shapefile") \
        .load(shapefile_path)

    # Perform transformations
    processed_df = df.selectExpr(
        "ST_Transform(geometry, 'EPSG:4326', 'EPSG:3857') as geometry",
        "attributes.*"
    )

    # Write to output (Parquet for efficient storage)
    processed_df.write \
        .mode("overwrite") \
        .parquet("s3a://processed-data/output.parquet")

    spark.stop()

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: spark_shapefile_processor.py <shapefile_path>",sys.argv[0])
        sys.exit(1)
    main(sys.argv[1])