import java.sql.Timestamp
import java.text.SimpleDateFormat
import java.util.Date
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkContext, SparkConf}

/**
  * wo xi huan xie dai ma
  * Created by wangtuntun on 16-5-7.
  */
object clean {

  def main(args: Array[String]) {

    //设置环境
    val conf=new SparkConf().setAppName("tianchi").setMaster("local")
    val sc=new SparkContext(conf)

    val songs=sc.textFile("/home/wangtuntun/AliMusic/Data/mars_tianchi_songs.csv")
    val user_actions=sc.textFile("/home/wangtuntun/AliMusic/Data/mars_tianchi_user_actions.csv")
    val songs_split=songs.map(_.split(","))
    val user_action_split=user_actions.map(_.split(","))
    val songs_pair=songs_split.map( x=>( (x(0)),(x(1),x(2),x(3),x(4),x(5)) ) )
    val user_action_pair=user_action_split.map( x=>((x(1)),(x(0),x(2),x(3),x(4)))  )
    val join=user_action_pair.join(songs_pair)
    //去掉播放时间小于出版时间的记录
    val filter=join.filter( x=>  (x._2._1._4  )<( x._2._2._2 ) )
    //将user_action里面的日期字符串转为数字序列（播放时间-出版日期）
    val start_date="20150301"
    val user_action_date_to_int=user_action_split.map{x=>
      val play_date=x(4)
      val days=play_date.toInt - start_date.toInt
      //(  x(0),x(1),x(2),x(3),days  )
       (x(0) + "," + x(1) +","+ x(2) +","+ x(3) +","+ days)

    }
    user_action_date_to_int.saveAsTextFile("/home/wangtuntun/AliMusic/Data/user_action_DS_int")
    sc.stop();

  }


}
