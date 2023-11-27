from pyspark import SparkConf
from pyspark.sql import SparkSession, Window
import pyspark.sql.types as t
import pyspark.sql.functions as f

spark = (
    SparkSession.builder
    .master('local')
    .appName('NYtaxi')
    .config(conf=SparkConf())
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")
spark.conf.set('spark.sql.legacy.timeParserPolicy', 'LEGACY')

pathData = 'data/trip_data_1.csv'
pathFare = 'data/trip_fare_12.csv'
resultPath = 'results/question'

taxiDataSchema = t.StructType([
    t.StructField('medallion', t.StringType(), True),
    t.StructField('hack_license', t.StringType(), True),
    t.StructField('vendor_id', t.StringType(), True),
    t.StructField('rate_code', t.IntegerType(), True),
    t.StructField('store_and_fwd_flag', t.StringType(), True),
    t.StructField('pickup_datetime', t.TimestampType(), True),
    t.StructField('dropoff_datetime', t.TimestampType(), True),
    t.StructField('passenger_count', t.IntegerType(), True),
    t.StructField('trip_time_in_secs', t.IntegerType(), True),
    t.StructField('trip_distance', t.FloatType(), True),
    t.StructField('pickup_longitude', t.DoubleType(), True),
    t.StructField('pickup_latitude', t.DoubleType(), True),
    t.StructField('dropoff_longitude', t.DoubleType(), True),
    t.StructField('dropoff_latitude', t.DoubleType(), True)
])

taxiFareSchema = t.StructType([
    t.StructField('medallion', t.StringType(), True),
    t.StructField('hack_license', t.StringType(), True),
    t.StructField('vendor_id', t.StringType(), True),
    t.StructField('pickup_datetime', t.TimestampType(), True),
    t.StructField('payment_type', t.StringType(), True),
    t.StructField('fare_amount', t.FloatType(), True),
    t.StructField('surcharge', t.FloatType(), True),
    t.StructField('mta_tax', t.FloatType(), True),
    t.StructField('tip_amount', t.FloatType(), True),
    t.StructField('tolls_amount', t.FloatType(), True),
    t.StructField('total_amount', t.FloatType(), True)
])
numericTypes = (t.IntegerType, t.FloatType, t.DoubleType)

taxiDf = spark.read.csv(pathData, header=True, nullValue='null', schema=taxiDataSchema)
taxiDf.show()
taxiDf.describe().show()
print(taxiDf.count())
taxiNumCols = [f.name for f in taxiDf.schema.fields if isinstance(f.dataType, numericTypes)]
taxiDf.select(taxiNumCols).summary().show()

fareDf = spark.read.csv(pathFare, header=True, nullValue='null', schema=taxiFareSchema)
fareDf.show()
fareDf.describe().show()
print(fareDf.count())
fareNumCols = [f.name for f in fareDf.schema.fields if isinstance(f.dataType, numericTypes)]
fareDf.select(fareNumCols).summary().show()

# What is average time for a trip per a weekday?
taxiDf = taxiDf.withColumn('day_of_week', f.date_format('pickup_datetime', 'u'))
dailyTripTime = (taxiDf.groupBy('day_of_week')
                 .agg(f.avg('trip_time_in_secs')
                      .alias('avg_trip_time')))
dailyTripTime.show()
dailyTripTime.write.csv(resultPath + '1', header=True)

# What is total revenue per medallion?
revenueDf = fareDf.select('medallion', 'total_amount')
totalRevenueDf = (revenueDf.groupBy('medallion')
                  .agg(f.sum('total_amount')
                       .alias('total_revenue')))
totalRevenueDf = totalRevenueDf.orderBy(f.desc('total_revenue'))
totalRevenueDf.show()
totalRevenueDf.write.csv(resultPath + '2', header=True)

# What is the tip amount compared to the fare amount for each taxi ride in percentage?
fareWindow = Window.partitionBy('pickup_datetime')
tipPercent = (fareDf.select('pickup_datetime', 'fare_amount', 'tip_amount')
              .withColumn('tip_percentage', (f.col('tip_amount') / f.col('fare_amount')) * 100))
tipPercent = tipPercent.filter(tipPercent.tip_percentage > 0)
tipPercent.show()
tipPercent.write.csv(resultPath + '3', header=True)

# What is the maximum number of passengers for each medallion?
passengerWindow = Window.partitionBy('medallion')
maxPassengers = (taxiDf.select('medallion', 'passenger_count')
                 .withColumn('max_passenger_count', f.max('passenger_count').over(passengerWindow))
                 .dropDuplicates(['medallion'])
                 .orderBy('medallion'))
maxPassengers = maxPassengers.drop(maxPassengers.passenger_count)
maxPassengers.show()
maxPassengers.write.csv(resultPath + '4', header=True)

joined_data = taxiDf.join(fareDf, ['medallion', 'hack_license', 'vendor_id'])

# What is average total amount for long distance trip per medallion?
fareForLongDistance = (joined_data.filter(joined_data.trip_distance > 10)
                       .groupBy('medallion')
                       .agg(f.avg('total_amount').alias('averagePayment')))
fareForLongDistance.show()
fareForLongDistance.write.csv(resultPath + '5', header=True)

# What is the average amount of tips that are left after trips with more than or equal to three passengers?
filtered_df = joined_data.filter(joined_data.passenger_count >= 3)
groupTips = filtered_df.agg(f.avg('tip_amount').alias('avg_tip_amount'))
groupTips.show()
groupTips.write.csv(resultPath + '6', header=True)
