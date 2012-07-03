/************************************************************************
  Copyright (c) 2003. David G. Lowe, University of British Columbia.
  This software is being made freely available for research purposes
  only (see file LICENSE.txt for conditions).  This notice must be
  retained on all copies.
*************************************************************************/

#ifndef _SIFT_KEY_H
#define _SIFT_KEY_H

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <vector>
#include <iostream>

namespace Lowe
{

class SIFT
{

  public:
    //-----------------
    // public typedefs
    //-----------------

    /** This structure describes a keypoint that has been found in an image. */
    struct Key
    {
      float row, col;        /* Row, column location relative to input image.  */
      float scale;           /* Scale (in sigma units for smaller DOG filter). */
      float ori;             /* Orientation in radians (-PI to PI). */
      float strength;        /* Added by David Gossow. Not part of the original algorithm */
      std::vector<int> ivec; /* Vector of gradient samples for indexing.*/
    };

    /** Typedef for a collection of keypoints */
    typedef std::vector< Key > KeyList;

    typedef std::vector< std::vector< float > >  FloatMatrix;

    typedef std::vector< std::vector< std::vector< float > > > Float3DArray;

    /** Data structure for a float image. */
    struct Image
    {
      /** Dimensions of image */
      int rows, cols;

      /** 2D array of image pixels */
      FloatMatrix pixels;

      /** Image constructor */
      Image ( int _rows = 0, int _cols = 0 )
      {
        setSize ( _rows, _cols );
      }

      /** Set the image size and allocate the vector. */
      void setSize ( int _rows, int _cols )
      {
        rows = _rows;
        cols = _cols;
        // allocate the image
        pixels.resize ( rows );
        for ( int r = 0;  r < rows;  ++r )
        {
          pixels[r].resize ( cols );
        }
      }
    };

    //-----------------
    // public methods
    //-----------------

    SIFT();

    /** Finds keypoints in a given image.
     *  @param[in] szImageFile  path to the input image file
     *  @return  the keypoint list */
    KeyList getKeypoints ( const char* szImageFile );

    KeyList getKeypoints ( const Image& image );

    /** Finds keypoints in a given image.
     *  @param[in] szImageFile  path to the input image file
     *  @param[out] rImage  an instance of the Image structure the image is read into
     *  @return  the keypoint list */
    KeyList getKeypoints ( const char* szImageFile, Image& rImage );

    /** Outputs keypoint data into a text file.
     *  @param[in] szOutputFile  output text file
     *  @param[in] keys  the keypoint list */
    void writeKeypoints ( const char* szOutputFile, const KeyList& keys );

    /** Visualizes keypoints by drawing arrows on the top of a given image.
     *  @param[in] im  image to draw onto (input image or a copy of it, normally)
     *  @param[in] keys  pointer to the beginning of the keypoint list */
    void drawKeypoints ( Image& im, const KeyList& keys );


    /**
     * Creates a Lowe::Image from raw char* data
     */
    Image fromRawData ( char* data, const int width, const int height );

    /** Magnitude of difference-of-Gaussian value at a keypoint must be
     *  above PeakThresh.  This avoids considering points with very low
     *  contrast that are dominated by noise.  PeakThreshInit is divided by
     *  Scales during initialization to give PeakThresh, because more
     *  closely spaced scale samples produce smaller DOG values.  A value
     *  of 0.08 considers only the most stable keypoints, but applications
     *  may wish to use lower values such as 0.02 to find keypoints from
     *  low-contast regions. */
    static float PeakThreshInit;

  protected:
    //---------------------------
    // protected constant fields
    //---------------------------

    //--------------------------------------------------------------------------
    // constants from the former util.c
    //--------------------------------------------------------------------------
    /** Value of PI, rounded up, so orientations are always in range [0,PI]. */

    /** These constants specify the size of the index vectors that provide
     *  a descriptor for each keypoint.  The region around the keypoint is
     *  sampled at OriSize orientations with IndexSize by IndexSize bins.
     *  VecLength is the length of the resulting index vector. */
    static const int OriSize;
    static const int IndexSize;
    static const int VecLength;

    static const double pi;

    /** This constant specifies how large a region is covered by each index
     *  vector bin.  It gives the spacing of index samples in terms of
     *  pixels at this scale (which is then multiplied by the scale of a
     *  keypoint).  It should be set experimentally to as small a value as
     *  possible to keep features local (good values are in range 3 to 5). */
    static const int MagFactor;

    /** Gaussian convolution kernels are truncated at this many sigmas from
     *  the center.  While it is more efficient to keep this value small,
     *  experiments show that for consistent scale-space analysis it needs
     *  a value at least 3.0, at which point the Gaussian has fallen to
     *  only 1% of its central value.  A value of 2.0 greatly reduces
     *  keypoint consistency, and a value of 4.0 is better than 3.0. */
    static const float GaussTruncate;


    //--------------------------------------------------------------------------
    // constants from the former key.c
    //--------------------------------------------------------------------------
    /** Set this constant to FALSE to skip step of doubling image size prior
     *  to finding keypoints.  This will reduce computation time by factor of 4
     *  but also find 4 times fewer keypoints. */
    static const bool DoubleImSize;

    /* Scales gives the number of discrete smoothing levels within each octave.
     * For example, Scales = 4 implies dividing octave into 4 intervals.
     * Value of 3 works well, but higher values find more keypoints. */
    static const int Scales;

    /** InitSigma gives the amount of smoothing applied to the image at the
     *  first level of each octave.  In effect, this determines the sampling
     *  needed in the image domain relative to amount of smoothing.  Good
     *  values determined experimentally are in the range 1.4 to 1.8. */
    static const float InitSigma;

    /** Peaks in the DOG function must be at least BorderDist samples away
     *  from the image border, at whatever sampling is used for that scale.
     *  Keypoints close to the border (BorderDist < about 15) will have part
     *  of the descriptor landing outside the image, which is approximated by
     *  having the closest image pixel replicated.  However, to perform as much
     *  matching as possible close to the edge, use BorderDist of 4. */
    static const int BorderDist;

    /** EdgeEigenRatio is used to eliminate keypoints that lie on an edge
     *  in the image without their position being accurately determined
     *  along the edge.  This can be determined by checking the ratio of
     *  eigenvalues of a Hessian matrix of the DOG function computed at the
     *  keypoint.  The eigenvalues are proportional to the two principle
     *  curvatures.  An EdgeEigenRatio of 10 means that all keypoints with
     *  a ratio of principle curvatures greater than 10 are discarded. */
    static const float EdgeEigenRatio;

    /* If UseHistogramOri flag is TRUE, then the histogram method is used
     * to determine keypoint orientation.  Otherwise, just use average
     * gradient direction within surrounding region (which has been found
     * to be less stable).  If using histogram, then OriBins gives the
     * number of bins in the histogram (36 gives 10 degree spacing of
     * bins). */
    static const bool UseHistogramOri;
    static const int OriBins;

    /** Size of Gaussian used to select orientations as multiple of scale
     *  of smaller Gaussian in DOG function used to find keypoint.
     *  Best values: 1.0 for UseHistogramOri = FALSE; 1.5 for TRUE. */
    static const float OriSigma;

    /** All local peaks in the orientation histogram are used to generate
     *  keypoints, as long as the local peak is within OriHistThresh of
     *  the maximum peak.  A value of 1.0 only selects a single orientation
     *  at each location. */
    static const float OriHistThresh;

    /** Width of Gaussian weighting window for index vector values.  It is
     *  given relative to half-width of index, so value of 1.0 means that
     *  weight has fallen to about half near corners of index patch.  A
     *  value of 1.0 works slightly better than large values (which are
     *  equivalent to not using weighting).  Value of 0.5 is considerably
     *  worse. */
    static const float IndexSigma;

    /** If this is TRUE, then treat gradients with opposite signs as being
     *  the same.  In theory, this could create more illumination invariance,
     *  but generally harms performance in practice. */
    static const bool IgnoreGradSign;

    /** Index values are thresholded at this value so that regions with
     *  high gradients do not need to match precisely in magnitude.
     *  Best value should be determined experimentally.  Value of 1.0
     *  has no effect.  Value of 0.2 is significantly better. */
    static const float MaxIndexVal;

    /** Set SkipInterp to TRUE to skip the quadratic fit for accurate peak
     *  interpolation within the pyramid.  This can be used to study the value
     *  versus cost of interpolation. */
    static const bool SkipInterp;

    //---------------------------
    // protected variable fields
    //---------------------------
    float PeakThresh;

    FloatMatrix H;  /**< Hessian matrix */
    bool m_bUninitializedH;  /**< true unless H is already initialized */

    //-------------------
    // protected methods
    //-------------------

    //--------------------------------------------------------------------------
    // methods from the former key.c
    //--------------------------------------------------------------------------

    /** Given an image, find the keypoints and return a pointer to a list of
     *  keypoint records. */
    KeyList GetKeypoints ( const Image& inputImage );

    /** Find keypoints within one octave of scale space starting with the
     *  given image.  The octSize parameter gives the size of each pixel
     *  relative to the input image.  Returns new list of keypoints after
     *  adding to the existing list "keys". */
    void OctaveKeypoints ( const Image& image, Image& nextImage, float octSize,
                           KeyList& keys );

    /** Find the local maxima and minima of the DOG images in scale space.
     *  Return the keypoints for these locations, added to existing "keys". */
    void FindMaxMin ( const Image* dogs, const Image* blur,
                      float octSize, KeyList& keys );

    /** Return TRUE iff val is a local maximum (positive value) or
     *  minimum (negative value) compared to the 3x3 neighbourhood that
     *  is centered at (row,col). */
    bool LocalMaxMin ( float val, const Image& dog, int row, int col );

    /** Returns FALSE if this point on the DOG function lies on an edge.
     *  This test is done early because it is very efficient and eliminates
     *  many points.  It requires that the ratio of the two principle
     *  curvatures of the DOG function at this point be below a threshold. */
    int NotOnEdge ( const Image& dog, int r, int c );

    /** Create a keypoint at a peak near scale space location (s,r,c), where
     *  s is scale (index of DOGs image), and (r,c) is (row, col) location.
     *  Return the list of keys with any new keys added. */
    void InterpKeyPoint ( const Image* dogs, int s, int r,
                          int c, const Image& grad, const Image& ori,
                          Image& map, float octSize,
                          KeyList& keys, int movesRemain );

    /** Assign an orientation to this keypoint.  This is done by creating a
     *  Gaussian weighted histogram of the gradient directions in the
     *  region.  The histogram is smoothed and the largest peak selected.
     *  The results are in the range of -PI to PI. */
    void AssignOriHist ( const Image& grad, const Image& ori, float octSize,
                         float octScale, float octRow, float octCol, float strength, KeyList& keys );

    /** Smooth a histogram by using a [1/3 1/3 1/3] kernel.  Assume the histogram
     *  is connected in a circular buffer. */
    void SmoothHistogram ( float* hist, int bins );

    /** Return a number in the range [-0.5, 0.5] that represents the
     *  location of the peak of a parabola passing through the 3 evenly
     *  spaced samples.  The center value is assumed to be greater than or
     *  equal to the other values if positive, or less than if negative. */
    float InterpPeak ( float a, float b, float c );

    /** Alternate approach to return an orientation for a keypoint, as
     *  described in the PhD thesis of Krystian Mikolajczyk.  This is done
     *  by creating a Gaussian weighted average of the gradient directions
     *  in the region.  The result is in the range of -PI to PI.  This was
     *  found not to work as well, but is included so that comparisons can
     *  continue to be done. */
    void AssignOriAvg ( const Image& grad, const Image& ori, float octSize,
                        float octScale, float octRow, float octCol, float strength, KeyList& keys );

    /** Create a new keypoint and return list of keypoints with new one added. */
    void MakeKeypoint ( const Image& grad, const Image& ori, float octSize,
                        float octScale, float octRow, float octCol, float angle, float strength,
                        KeyList& keys );

    /*--------------------- Making image descriptor -------------------------*/

    /** Use the parameters of this keypoint to sample the gradient images
     *  at a set of locations within a circular region around the keypoint.
     *  The (scale,row,col) values are relative to current octave sampling.
     *  The resulting vector is stored in the key. */
    void MakeKeypointSample ( Key& key, const Image& grad,
                              const Image& ori, float scale, float row, float col );

    /* Normalize length of vec to 1.0. */
    void NormalizeVec ( float* vec, int len );

    /** Create a 3D index array into which gradient values are accumulated.
     *  After filling array, copy values back into vec. */
    void KeySampleVec ( float *vec, const Key& key, const Image& grad,
                        const Image& ori, float scale, float row, float col );

    /** Add features to vec obtained from sampling the grad and ori images
     *  for a particular scale.  Location of key is (scale,row,col) with respect
     *  to images at this scale.  We examine each pixel within a circular
     *  region containing the keypoint, and distribute the gradient for that
     *  pixel into the appropriate bins of the index array. */
    void KeySample ( Float3DArray& index,
                     const Key& key, const Image& grad, const Image& ori,
                     float scale, float row, float col );

    /* Given a sample from the image gradient, place it in the index array. */
    void AddSample ( Float3DArray& index,
                     const Key& k, const Image& grad, const Image& orim,
                     int r, int c, float rpos, float cpos,
                     float rx, float cx );

    /** Increment the appropriate locations in the index to incorporate
     * this image sample.  The location of the sample in the index is (rx,cx). */
    void PlaceInIndex ( Float3DArray& index,
                        float mag, float ori, float rx, float cx );

    /** Draw the given line in the image, but first translate, rotate, and
     *  scale according to the keypoint parameters. */
    void TransformLine ( Image& im, const Key& k,
                         float x1, float y1, float x2, float y2 );

    /** Write set of keypoints to a file in ASCII format.
     *  The file format starts with 2 integers giving the total number of
     *  keypoints, and size of descriptor vector for each keypoint. Then
     *  each keypoint is specified by 4 floating point numbers giving
     *  subpixel row and column location, scale, and orientation (in
     *  radians from -PI to PI).  Then the descriptor vector for each
     *  keypoint is given as a list of integers in range [0,255]. */
    void WriteKeypoints ( FILE *fp, const KeyList& keys );

    //--------------------------------------------------------------------------
    // methods from the former util.c
    //--------------------------------------------------------------------------
    template< typename T >
    T LOWE_ABS ( T x )
    {
      return ( x > 0 ) ? ( x ) : ( -x );
    }

    template< typename T >
    T LOWE_MAX ( T x, T y )
    {
      return ( x > y ) ? ( x ) : ( y );
    }

    template< typename T >
    T LOWE_MIN ( T x, T y )
    {
      return ( x < y ) ? ( x ) : ( y );
    }

    FloatMatrix AllocMatrix ( int rows, int cols );

    /*----------------------- Image utility routines ----------------------*/

    /** Double image size. Use linear interpolation between closest pixels.
     *  Size is two rows and columns short of double to simplify interpolation. */
    Image DoubleSize ( const Image& image );

    /** Reduce the size of the image by half by selecting alternate pixels on
     *  every row and column.  We assume image has already been blurred
     *  enough to avoid aliasing. */
    Image HalfImageSize ( const Image& image );

    /* Subtract image im2 from im1 and leave result in im1. */
    void SubtractImage ( Image& im1, const Image& im2 );

    /** Given a smoothed image, im, return image gradient and orientation
     *  at each pixel in grad and ori.  Note that since we will eventually
     *  be normalizing gradients, we only need to compute their relative
     *  magnitude within a scale image (so no need to worry about change in
     *  pixel sampling relative to sigma or other scaling issues). */
    void GradOriImages ( const Image& im, Image& grad, Image& ori );

    /* --------------------------- Blur image --------------------------- */

    /** Convolve image with a Gaussian of width sigma and store result back
     *  in image.   This routine creates the Gaussian kernel, and then applies
     *  it sequentially in horizontal and vertical directions. */
    void GaussianBlur ( Image& image, float sigma );

    /** Convolve image with the 1-D kernel vector along image rows.  This
     *  is designed to be as efficient as possible.  Pixels outside the
     *  image are set to the value of the closest image pixel. */
    void ConvHorizontal ( Image& image, float *kernel, int ksize );

    /* Same as ConvHorizontal, but apply to vertical columns of image. */
    void ConvVertical ( Image& image, float *kernel, int ksize );

    /** Perform convolution of the kernel with the buffer, returning the
     *  result at the beginning of the buffer.  rsize is the size
     *  of the result, which is buffer size - ksize + 1. */
    void ConvBuffer ( float *buffer, float *kernel, int rsize, int ksize );

    /** Same as ConvBuffer, but implemented with loop unrolling for increased
     *  speed.  This is the most time intensive routine in keypoint detection,
     *  so deserves careful attention to efficiency.  Loop unrolling simply
     *  sums 5 multiplications at a time to allow the compiler to schedule
     *  operations better and avoid loop overhead.  This almost triples
     *  speed of previous version on a Pentium with gcc. */
    void ConvBufferFast ( float *buffer, float *kernel, int rsize, int ksize );

    /*--------------------- Least-squares solutions ---------------------------*/

    /** Give a least-squares solution to the system of linear equations given in
     *  the jacobian and errvec arrays.  Return result in solution.
     *  This uses the method of solving the normal equations. */
    void SolveLeastSquares ( float *solution, int rows, int cols,
                             const FloatMatrix& jacobian,
                             float *errvec, FloatMatrix& sqarray );

    /** Solve the square system of linear equations, Ax=b, where A is given
     *  in matrix "sq" and b in the vector "solution".  Result is given in
     *  solution.  Uses Gaussian elimination with pivoting. */
    void SolveLinearSystem ( float *solution, FloatMatrix& sq, int size );

    /* Return dot product of two vectors with given length. */
    float DotProd ( float *v1, float *v2, int len );

    /** Apply the method developed by Matthew Brown (see BMVC 02 paper) to
     *  fit a 3D quadratic function through the DOG function values around
     *  the location (s,r,c), i.e., (scale,row,col), at which a peak has
     *  been detected.  Return the interpolated peak position by setting
     *  the vector "offset", which gives offset from position (s,r,c).  The
     *  returned function value is the interpolated DOG magnitude at this peak. */
    float FitQuadratic ( float offset[3], const Image* dogs, int s, int r, int c );

    /** Read a PGM file from the given file pointer and return it as a
     *  floating point SIFT::Image structure.  See "man pgm" for details on PGM
     *  file format.  This handles only the usual 8-bit "raw" PGM format.
     *  Use xv or the PNM tools (such as pnmdepth) to convert from other
     *  formats. */
    Image ReadPGM ( FILE *fp );

    /** Skip white space including any comments. PGM files allow a comment
     *  starting with '#' to end-of-line. */
    void SkipComments ( FILE *fp );

    /* Write an SIFT::Image in PGM format to the file fp. */
    void WritePGM ( FILE *fp, const Image& image );

    /** Draw a white line from (r1,c1) to (r2,c2) on the image.  Both points
     *  must lie within the image. */
    void DrawLine ( Image& image, int r1, int c1, int r2, int c2 );
};

}

#endif

