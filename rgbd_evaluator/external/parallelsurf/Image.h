/*
* This file is part of Parallel SURF, which implements the SURF algorithm
* using multi-threading.
*
* Copyright (C) 2010 David Gossow
*
* It is based on the SURF implementation included in Pan-o-matic 0.9.4,
* written by Anael Orlinski.
*
* Parallel SURF is free software; you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation; either version 3 of the License, or
* (at your option) any later version.
*
* Parallel SURF is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef __parallelsurf_image_h
#define __parallelsurf_image_h

namespace parallelsurf
{

// forward declaration
class IntegralImage;

class Image
{
	public:
		Image() : _ii ( 0 ), _width ( 0 ), _height ( 0 ) {};

		// Constructor from a pixel array
		Image ( const unsigned char **iPixels, unsigned int iWidth, unsigned int iHeight );

		// setup the integral image
		void init ( const unsigned char **iPixels, unsigned int iWidth, unsigned int iHeight );

		// cleanup
		void clean();

		// Destructor
		~Image();

		// Accessors
		inline const unsigned char **		getPixels()
		{
			return _pixels;
		}
		inline double **		getIntegralImage()
		{
			return _ii;
		}
		inline unsigned int		getWidth()
		{
			return _width;
		}
		inline unsigned int		getHeight()
		{
			return _height;
		}

		// allocate and deallocate integral image pixels
		static double **		AllocateImage ( unsigned int iWidth, unsigned int iHeight );
		static void				DeallocateImage ( double **iImagePtr, unsigned int iHeight );
		
		static void setDoRandomInit( bool iDoRandomInit=true );
		static bool getDoRandomInit() { return DoRandomInit; }

	private:

		// prepare the integral image
		void					buildIntegralImage();

		// pixel data of the image
		const unsigned char**				_pixels;

		// integral image
		double**        _ii; // Data of the integral image Like data[lines][rows]

		// image size
		unsigned int			_width;
		unsigned int			_height;
		
		static bool DoRandomInit;
};

}

#endif //__parallelsurf_image_h

