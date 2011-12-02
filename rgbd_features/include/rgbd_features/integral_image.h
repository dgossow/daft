/*
* Copyright (C) 2011 David Gossow
*/

#ifndef __rgbd_features_image_h
#define __rgbd_features_image_h

namespace rgbd_features
{

class IntegralImage
{
public:
    IntegralImage() : _ii ( 0 ), _width ( 0 ), _height ( 0 ) {};

    // Constructor from a pixel array
    IntegralImage ( const unsigned char **iPixels, unsigned int iWidth, unsigned int iHeight );

    // allocate the integral image
    void init ( unsigned int iWidth, unsigned int iHeight );

    // compute the integral image
    void computeFrom( const unsigned char** iPixels );

    // cleanup
    void clean();

    // Destructor
    ~IntegralImage();

    inline double ** getIntegralImage()
    {
        return _ii;
    }
    inline unsigned int getWidth()
    {
        return _width;
    }
    inline unsigned int getHeight()
    {
        return _height;
    }

    // allocate and deallocate integral image pixels
    static double ** AllocateImage ( unsigned int iWidth, unsigned int iHeight );
    static void DeallocateImage ( double **iImagePtr, unsigned int iHeight );

private:

    // integral image
    double** _ii;

    // image size
    unsigned int _width;
    unsigned int _height;
};

}

#endif //__rgbd_features_image_h

