

static std::map< std::string, std::pair<double, double> > __timing_list;

#define TIME_MS( FUNCTION ) \
{ \
  timeval t1,t2; \
  gettimeofday(&t1, NULL); \
  FUNCTION \
  gettimeofday(&t2, NULL); \
  double elapsed_time = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0; \
/*  ROS_INFO( "%.4lf ms for " #FUNCTION, elapsed_time ); */ \
  \
  if ( __timing_list.find( #FUNCTION ) == __timing_list.end() ) { \
    __timing_list[ #FUNCTION ] = std::pair<double,double>(0,0); \
  } \
  __timing_list[ #FUNCTION ].second += elapsed_time; \
}

void increase_counters()
{
  std::map< std::string, std::pair<double, double> >::iterator timing_it = __timing_list.begin();
  while( timing_it != __timing_list.end() )
  {
    timing_it->second.first ++;
    timing_it++;
  }
}

void print_time()
{
  double time_total = 0;
  std::map< std::string, std::pair<double, double> >::iterator timing_it = __timing_list.begin();
  while( timing_it != __timing_list.end() )
  {
    double time_mean = timing_it->second.second/timing_it->second.first;
    ROS_INFO( "%.4lf ms mean time for %s", time_mean, timing_it->first.c_str() );
    time_total += time_mean;
    timing_it++;
  }
  ROS_INFO( "%.4lf ms mean time in total.", time_total );
}
