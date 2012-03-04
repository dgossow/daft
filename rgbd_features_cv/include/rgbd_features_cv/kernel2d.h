
#ifndef rgbd_features_kernel2d_h_
#define rgbd_features_kernel2d_h_

namespace cv
{

inline void showBig( int size, cv::Mat img, std::string name )
{
  cv::Mat1f img_big;
  cv::resize( img, img_big, cv::Size(size,size), 0, 0, INTER_NEAREST );
  cv::imshow( name, img_big );
  cv::waitKey(100);
}

struct Kernel2D
{
  static inline float Gaussian2(float sigma, float d2) {
    return 0.39894228f / sigma * std::exp(-0.5f * d2 / (sigma * sigma));
  }

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

  float kernel[9][9];
};

struct LaplaceKernel: public Kernel2D
{
  static inline float LoG2(float sigma, float d2) {
    float s2 = sigma * sigma;
    float s4 = s2 * s2;
    float arg = 0.5f * d2 / s2;
    return (arg - 1.0f) / (3.1415f * s4) * std::exp(-arg);
  }

  LaplaceKernel() {
    for(int i=0; i<9; i++) {
      for(int j=0; j<9; j++) {
        float d2 = (float(i)-4)*(float(i)-4) + (float(j)-4)*(float(j)-4);
        kernel[i][j] = LoG2(1.2f, d2);
      }
    }
  }

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

  float kernel[9][9];
};

static LaplaceKernel sLaplaceKernel;


struct DxxKernel: public Kernel2D
{
  DxxKernel() {
    for(int i=0; i<9; i++) {
      const int i2 = i-4;
      for(int j=0; j<9; j++) {
        const int j2 = j-4;
        float d2 = (float(i2))*(float(i2)) + (float(j2))*(float(j2));
        static const float sigma = 1.2f;
        static const float sigma2 = sigma*sigma;
        static const float sigma4 = sigma2*sigma2;
        kernel[i][j] = Gaussian2(1.2f, d2) * ( j2*j2/sigma4 - 1/sigma2 );
      }
    }
  }
};
struct DyyKernel: public Kernel2D
{
  DyyKernel() {
    for(int i=0; i<9; i++) {
      const int i2 = i-4;
      for(int j=0; j<9; j++) {
        const int j2 = j-4;
        float d2 = (float(i2))*(float(i2)) + (float(j2))*(float(j2));
        static const float sigma = 1.2f;
        static const float sigma2 = sigma*sigma;
        static const float sigma4 = sigma2*sigma2;
        kernel[i][j] = Gaussian2(1.2f, d2) * ( i2*i2/sigma4 - 1/sigma2 );
      }
    }
  }
};
struct DxyKernel: public Kernel2D
{
  DxyKernel() {
    for(int i=0; i<9; i++) {
      const int i2 = i-4;
      for(int j=0; j<9; j++) {
        const int j2 = j-4;
        float d2 = (float(i2))*(float(i2)) + (float(j2))*(float(j2));
        static const float sigma = 1.2f;
        static const float sigma2 = sigma*sigma;
        static const float sigma4 = sigma2*sigma2;
        kernel[i][j] = Gaussian2(1.2f, d2) * ( i2*j2/sigma4 );
      }
    }
  }
};

static DxxKernel sDxxKernel;
static DyyKernel sDyyKernel;
static DxyKernel sDxyKernel;


}

#endif
