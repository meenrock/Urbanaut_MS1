from pyspark.sql import SparkSession
from sedona.register import SedonaRegistrator
from sedona.utils import SedonaKryoRegistrator

class ShapeFileInputSpark:

    def read_shape_file(self, filepath):
        spark = SparkSession.builder \
            .appName("ShapefileWithSedona") \
            .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
            .config("spark.kryo.registrator", "org.apache.sedona.core.serde.SedonaKryoRegistrator") \
            .config("spark.kryoserializer.buffer.max", "1024m") \
            .config("spark.kryoserializer.buffer", "256k") \
            .getOrCreate()

        SedonaRegistrator.registerAll(spark)

        shapefile_path = "path/to/your/file.shp"
        spark_df = spark.read \
            .format("shapefile") \
            .option("reduceSplit", "true") \
            .load(shapefile_path)

        spark_df.cache()

        spark_df.show(5)

