
#ifndef rgbd_features_kernel2d_h_
#define rgbd_features_kernel2d_h_

namespace cv
{

inline void showBig( int size, cv::Mat img, std::string name )
{
  cv::Mat1f img_big;
  if(size > 0) {
    cv::resize( img, img_big, cv::Size(size,size), 0, 0, INTER_NEAREST );
  }
  else {
    cv::resize( img, img_big, cv::Size(0,0), 4, 4, INTER_NEAREST );
  }
  cv::imshow( name, img_big );
  cv::waitKey(100);
}

namespace detail
{
  inline float Gaussian2(float sigma, float d2) {
    return 0.39894228f / sigma * std::exp(-0.5f * d2 / (sigma * sigma));
  }

  inline float LoG2(float sigma, float x, float y) {
    float s2 = sigma * sigma;
    float s4 = s2 * s2;
    float arg = 0.5f * (x*x + y*y) / s2;
    return (arg - 1.0f) / (3.1415f * s4) * std::exp(-arg);
  }

  inline float DxxKernel(float sigma, float x, float y) {
    const float sigma2 = sigma*sigma;
    const float sigma4 = sigma2*sigma2;
    return Gaussian2(sigma, x*x + y*y) * ( x*x/sigma4 - 1/sigma2 );
  }

  inline float DyyKernel(float sigma, float x, float y) {
    const float sigma2 = sigma*sigma;
    const float sigma4 = sigma2*sigma2;
    return Gaussian2(sigma, x*x + y*y) * ( y*y/sigma4 - 1/sigma2 );
  }

  inline float DxyKernel(float sigma, float x, float y) {
    const float sigma2 = sigma*sigma;
    const float sigma4 = sigma2*sigma2;
    return Gaussian2(sigma, x*x + y*y) * ( x*y/sigma4 );
  }

}

struct Kernel2D
{
  float convolve(float values[9][9]) const {
    float sum = 0.0f;
    for(int i=0; i<9; i++) {
      for(int j=0; j<9; j++) {
        sum += kernel[i][j] * values[i][j];
      }
    }
    return sum;
  }

  cv::Mat1f asCvImage() const {
    cv::Mat1f img(9,9);
    for(int i=0; i<9; i++) {
      for(int j=0; j<9; j++) {
        img[i][j] = kernel[i][j];
      }
    }
    return img;
  }

  template <float (*F)(float,float,float), int Samples>
  void create(float sigma) {
    for(int i=0; i<9; i++) {
      float y = float(i-4);
      for(int j=0; j<9; j++) {
        float x = float(j-4);
        float v;
        if(Samples > 0) {
          v = 0.0f;
          for(int i=-Samples; i<=+Samples; i++) {
            for(int j=-Samples; j<=+Samples; j++) {
              v += F(sigma, x + float(j)*float(Samples)/float(2*Samples+1) , y + float(i)*float(Samples)/float(2*Samples+1));
            }
          }
          v /= float((2*Samples + 1)*(2*Samples + 1));
        }
        else {
          v = F(sigma, x, y);
        }
        kernel[i][j] = v;
      }
    }
  }

  template <float (*F)(float,float,float), bool AxisAligned, int Samples>
  void create(const Matx22f& rotation, float ratio, float sigma) {
//    float sum_pos = 0.0f;
//    float sum_neg = 0.0f;
    for(int i=0; i<9; i++) {
      float y = float(i-4);
      for(int j=0; j<9; j++) {
        float x = float(j-4);
        Point2f q = rotation * Point2f(x,y);
        q.y /= ratio;
        if(AxisAligned) {
          q = rotation.t() * q;
        }
        float v;
        if(Samples > 0) {
          v = 0.0f;
          for(int i=-Samples; i<=+Samples; i++) {
            for(int j=-Samples; j<=+Samples; j++) {
              v += F(sigma, q.x + float(j)*float(Samples)/float(2*Samples+1) , q.y + float(i)*float(Samples)/float(2*Samples+1));
            }
          }
          v /= float((2*Samples + 1)*(2*Samples + 1));
        }
        else {
          v = F(sigma, q.x, q.y);
        }
        kernel[i][j] = v / ratio;
//        if(v >= 0.0f) {
//          sum_pos += v;
//        }
//        else {
//          sum_neg += v;
//        }
      }
    }
//    std::cout << "pos=" <<  sum_pos << " " << sum_neg << std::endl;
//    for(int i=0; i<9; i++) {
//      for(int j=0; j<9; j++) {
//        float& v = kernel[i][j];
//        if(v >= 0.0f) {
//          if(sum_pos != 0) {
//            v /= sum_pos;
//          }
//        }
//        else {
//          if(sum_pos != 0) {
//            v /= -sum_neg;
//          }
//        }
//      }
//    }
  }

  template <float (*F)(float,float,float), int Samples>
  static Kernel2D Create(float sigma) {
    Kernel2D q;
    q.create<F,Samples>(sigma);
    return q;
  }

  template <float (*F)(float,float,float), bool AxisAligned, int Samples>
  static Kernel2D Create(const Matx22f& rotation, float ratio, float sigma) {
    Kernel2D q;
    q.create<F,AxisAligned,Samples>(rotation, ratio, sigma);
    return q;
  }

  float kernel[9][9];
};

static Kernel2D sLaplaceKernel = Kernel2D::Create<detail::LoG2,1>(1.2f);
static Kernel2D sDxxKernel = Kernel2D::Create<detail::DxxKernel,1>(1.2f);
static Kernel2D sDyyKernel = Kernel2D::Create<detail::DyyKernel,1>(1.2f);
static Kernel2D sDxyKernel = Kernel2D::Create<detail::DxyKernel,1>(1.2f);


struct Kernel2DCache
{
  static const float cRatioMin = 0.25f;
  static const float cRatioMax = 1.0f;
  static const int cRatioSteps = 20;
  static const float cAngleMin = 0.0f;
  static const float cAngleMax = 3.1415f;
  static const int cAngleSteps = 30;

  float convolve(float values[9][9], float ratio, float angle) const {
    int ox = findRatioPos(ratio);
    int oy = findAnglePos(angle);
    return convolve_impl(values, ox, oy);
  }

  template <float (*F)(float,float,float), bool AxisAligned, int Samples>
  static Kernel2DCache Create(float sigma, const char* name) {
    Kernel2DCache q;
    q.create<F,AxisAligned,Samples>(sigma, name);
    return q;
  }

private:
  template <float (*F)(float,float,float), bool AxisAligned, int Samples>
  void create(float sigma, const char* name) {
    kernel_cache_ = Mat1f(9*(cAngleSteps+1), 9*(cRatioSteps+1));
    for(int i=0; i<cAngleSteps+1; i++) {
      for(int j=0; j<cRatioSteps+1; j++) {
        float angle = cAngleMin + float(i) / float(cAngleSteps) * (cAngleMax - cAngleMin);
        float ratio = cRatioMin + float(j) / float(cRatioSteps) * (cRatioMax - cRatioMin);
//        std::cout << angle << " " << ratio << std::endl;
        float ca = std::cos(angle);
        float sa = std::sin(angle);
        Matx22f rotation;
        rotation(0,0) = ca;
        rotation(0,1) = sa;
        rotation(1,0) = -sa;
        rotation(1,1) = ca;
        rotation = rotation.t();
//        std::cout << rotation(0,0) << " " << rotation(0,1) << " " << rotation(1,0) << " " << rotation(1,1) << " " << std::endl;
        Kernel2D k2 = Kernel2D::Create<F,AxisAligned,Samples>(rotation, ratio, sigma);
//        Kernel2D k2 = Kernel2D::Create<F,Samples>(sigma);
        for(int i2=0; i2<9; i2++) {
          for(int j2=0; j2<9; j2++) {
            kernel_cache_[9*i + i2][9*j + j2] = k2.kernel[i2][j2];
          }
        }
      }
    }
//    showBig(0, kernel_cache_ + 0.5f, name);
  }

  int findRatioPos(float ratio) const {
    float p = (ratio - cRatioMin) / (cRatioMax - cRatioMin) * float(cRatioSteps + 1);
    int pi = static_cast<int>(p + 0.5f); // round
    if(pi < 0) {
      pi = 0;
    }
    if(pi >= cRatioSteps + 1) {
      pi = cRatioSteps;
    }
    return pi;
  }

  int findAnglePos(float angle) const {
    while(angle < 0.0f) angle += M_2_PI;
    while(angle > M_2_PI) angle -= M_2_PI;
    if(angle > M_PI) angle -= M_PI;
    float p = (angle - cAngleMin) / (cAngleMax - cAngleMin) * float(cAngleSteps + 1);
    int pi = static_cast<int>(p + 0.5f); // round
    if(pi == cAngleSteps + 1) {
      pi --;
    }
    return pi;
  }

  float convolve_impl(float values[9][9], int ox, int oy) const {
    float sum = 0.0f;
    for(int i=0; i<9; i++) {
      for(int j=0; j<9; j++) {
        sum += kernel_cache_[9*oy+i][9*ox+j] * values[i][j];
      }
    }
    return sum;
  }

private:
  Mat1f kernel_cache_;

};


static Kernel2DCache sLaplaceKernelCache = Kernel2DCache::Create<detail::LoG2, false, 1>(1.2f, "LoG2");
static Kernel2DCache sDxxKernelCache = Kernel2DCache::Create<detail::DxxKernel, false, 1>(1.2f, "DxxKernel");
static Kernel2DCache sDyyKernelCache = Kernel2DCache::Create<detail::DyyKernel, false, 1>(1.2f, "DyyKernel");
static Kernel2DCache sDxyKernelCache = Kernel2DCache::Create<detail::DxyKernel, false, 1>(1.2f, "DxyKernel");
//static Kernel2DCache sLaplaceKernelCacheNaa = Kernel2DCache::Create<detail::LoG2, false, 1>(1.2f, "LoG2 NAA");
//static Kernel2DCache sDxxKernelCacheNaa = Kernel2DCache::Create<detail::DxxKernel, false, 1>(1.2f, "DxxKernel NAA");
//static Kernel2DCache sDyyKernelCacheNaa = Kernel2DCache::Create<detail::DyyKernel, false, 1>(1.2f, "DyyKernel NAA");
//static Kernel2DCache sDxyKernelCacheNaa = Kernel2DCache::Create<detail::DxyKernel, false, 1>(1.2f, "DxyKernel NAA");

}

#endif
