package lsd

/*----------------------------------------------------------------------------

  LSD - Line Segment Detector on digital images

  Copyright 2007-2011 rafael grompone von gioi (grompone@gmail.com)

  This program is free software: you can redistribute it and/or modify
  it under the terms of the GNU Affero General Public License as
  published by the Free Software Foundation, either version 3 of the
  License, or (at your option) any later version.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
  GNU Affero General Public License for more details.

  You should have received a copy of the GNU Affero General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.

  ----------------------------------------------------------------------------*/

/*----------------------------------------------------------------------------*/
/** @file lsd.c
  LSD module code
  @author rafael grompone von gioi (grompone@gmail.com)
*/
/*----------------------------------------------------------------------------*/

/*----------------------------------------------------------------------------*/
/** @mainpage LSD code documentation

  This is an implementation of the Line Segment Detector described
  in the paper:

    "LSD: A Fast Line Segment Detector with a False Detection Control"
    by Rafael Grompone von Gioi, Jeremie Jakubowicz, Jean-Michel Morel,
    and Gregory Randall, IEEE Transactions on Pattern Analysis and
    Machine Intelligence, vol. 32, no. 4, pp. 722-732, April, 2010.

  and in more details in the CMLA Technical Report:

    "LSD: A Line Segment Detector, Technical Report",
    by Rafael Grompone von Gioi, Jeremie Jakubowicz, Jean-Michel Morel,
    Gregory Randall, CMLA, ENS Cachan, 2010.

  The version implemented here includes some further improvements
  described on the LSD page at www.ipol.im. That same page includes
  more information, including this code and an online demo version:

    http://www.ipol.im/pub/algo/gjmr_line_segment_detector

  The module's main function is LSD().

  The source code is contained in two files: lsd.go

  HISTORY:
  - version 0.1 - may 2019: Initial Go version

  - version 1.6 - nov 2011: Changes in the interface,
                            max_grad parameter removed,
                            the factor 11 was added to the number of test
                            to consider the different precision values tested,
                            a minor bug corrected in the gradient sorting
                            code.
  - version 1.5 - dec 2010: Changes in 'refine', -W option added,
                            and more comments added.
  - version 1.4 - jul 2010: lsd_scale interface added and doxygen doc.
  - version 1.3 - feb 2010: Multiple bug correction and improved code.
  - version 1.2 - dec 2009: First full Ansi C Language version.
  - version 1.1 - sep 2009: Systematic subsampling to scale 0.8 and
                            correction to partially handle"angle problem".
  - version 1.0 - jan 2009: First complete Megawave2 and Ansi C Language
                            version.

  @author rafael grompone von gioi (grompone@gmail.com)
*/
/*----------------------------------------------------------------------------*/

import (
	"log"
	"math"
	"sort"
)

const (
	/** Label for pixels with undefined gradient. */
	NOTDEF = -1024.0

	/** 3/2 pi */
	M_3_2_PI = math.Pi * 3.0 / 2.0

	/** 2 pi */
	M_2__PI = math.Pi * 2

	/** Doubles relative error factor */
	RELATIVE_ERROR_FACTOR = 100.0

	DBL_MIN = math.SmallestNonzeroFloat64
	DBL_MAX = math.MaxFloat64
)

var (
	DBL_EPSILON = math.Nextafter(1.0, 1.1) - 1.0
)

/*----------------------------------------------------------------------------*/
/** Chained list of coordinates.
 */
type coorlist struct {
	x, y int
	next *coorlist
}

/*----------------------------------------------------------------------------*/
/** A point (or pixel).
 */
type point struct {
	x, y int
}

/*----------------------------------------------------------------------------*/
/*------------------------- Miscellaneous functions --------------------------*/
/*----------------------------------------------------------------------------*/

/*----------------------------------------------------------------------------*/
/** Fatal error, print a message to standard-error output and exit.
 */
func fatal_error(msg string, params ...interface{}) {
	log.Fatalf("LSD Error: "+msg, params...)
}

/*----------------------------------------------------------------------------*/
/** Compare doubles by relative error.

  The resulting rounding error after floating point computations
  depend on the specific operations done. The same number computed by
  different algorithms could present different rounding errors. For a
  useful comparison, an estimation of the relative rounding error
  should be considered and compared to a factor times EPS. The factor
  should be related to the cumulated rounding error in the chain of
  computation. Here, as a simplification, a fixed factor is used.
*/
func double_equal(a, b float64) bool {
	/* trivial case */
	if a == b {
		return true
	}

	abs_diff := math.Abs(a - b)
	aa := math.Abs(a)
	bb := math.Abs(b)

	abs_max := bb
	if aa > bb {
		abs_max = aa
	}

	/* DBL_MIN is the smallest normalized number, thus, the smallest
	   number whose relative error is bounded by DBL_EPSILON. For
	   smaller numbers, the same quantization steps as for DBL_MIN
	   are used. Then, for smaller numbers, a meaningful "relative"
	   error should be computed by dividing the difference by DBL_MIN. */
	if abs_max < DBL_MIN {
		abs_max = DBL_MIN
	}

	/* equal if relative error <= factor x eps */
	return (abs_diff / abs_max) <= (RELATIVE_ERROR_FACTOR * DBL_EPSILON)
}

/*----------------------------------------------------------------------------*/
/** Computes Euclidean distance between point (x1,y1) and point (x2,y2).
 */
func dist(x1, y1, x2, y2 float64) float64 {
	return math.Sqrt((x2-x1)*(x2-x1) + (y2-y1)*(y2-y1))
}

/*----------------------------------------------------------------------------*/
/*----------------------- 'list of n-tuple' data type ------------------------*/
/*----------------------------------------------------------------------------*/

/*----------------------------------------------------------------------------*/
/** 'list of n-tuple' data type

  The i component, of the n-tuple number j, of an n-tuple list 'ntl'
  is accessed with:

    ntl.values[ i + j * ntl.dim ]

  The dimension of the n-tuple (n) is:

    ntl.dim

  The number of n-tuples in the list is:

    ntl.size
*/
type ntuple_list struct {
	size   uint
	dim    uint
	values []float64
}

/*----------------------------------------------------------------------------*/
/** Create an n-tuple list and allocate memory for one element.
  @param dim the dimension (n) of the n-tuple.
*/
func new_ntuple_list(dim uint) *ntuple_list {
	/* check parameters */
	if dim == 0 {
		fatal_error("new_ntuple_list: 'dim' must be positive.")
	}

	/* initialize list */
	return &ntuple_list{size: 0, dim: dim}
}

/*----------------------------------------------------------------------------*/
/** Add a 7-tuple to an n-tuple list.
 */
func add_7tuple(out *ntuple_list, v1, v2, v3, v4, v5, v6, v7 float64) {
	/* check parameters */
	if out == nil {
		fatal_error("add_7tuple: invalid n-tuple input.")
	}
	if out.dim != 7 {
		fatal_error("add_7tuple: the n-tuple must be a 7-tuple.")
	}

	/* add new 7-tuple */
	out.values = append(out.values, v1)
	out.values = append(out.values, v2)
	out.values = append(out.values, v3)
	out.values = append(out.values, v4)
	out.values = append(out.values, v5)
	out.values = append(out.values, v6)
	out.values = append(out.values, v7)

	/* update number of tuples counter */
	out.size++
}

/*----------------------------------------------------------------------------*/
/*----------------------------- Image Data Types -----------------------------*/
/*----------------------------------------------------------------------------*/

/*----------------------------------------------------------------------------*/
/** char image data type

  The pixel value at (x,y) is accessed by:

    image.data[ x + y * image.xsize ]

  with x and y integer.
*/
type image_bool struct {
	data         []bool
	xsize, ysize uint
}

/*----------------------------------------------------------------------------*/
/** Create a new image_bool of size 'xsize' times 'ysize'.
 */
func new_image_bool(xsize, ysize uint) *image_bool {
	/* check parameters */
	if xsize == 0 || ysize == 0 {
		fatal_error("new_image_bool: invalid image size.")
	}

	image := image_bool{xsize: xsize, ysize: ysize, data: make([]bool, xsize*ysize)}

	/* set image size */
	return &image
}

/*----------------------------------------------------------------------------*/
/** Create a new image_bool of size 'xsize' times 'ysize',
  initialized to the value 'fill_value'.
*/
func new_image_bool_ini(xsize, ysize uint, fill_value bool) *image_bool {
	image := new_image_bool(xsize, ysize) /* create image */

	/* initialize */
	for i := 0; i < len(image.data); i++ {
		image.data[i] = fill_value
	}

	return image
}

/*----------------------------------------------------------------------------*/
/** int image data type

  The pixel value at (x,y) is accessed by:

    image.data[ x + y * image.xsize ]

  with x and y integer.
*/
type image_int struct {
	xsize, ysize uint
	data         []int
}

/*----------------------------------------------------------------------------*/

/** Create a new image_int of size 'xsize' times 'ysize'.
 */
func new_image_int(xsize, ysize uint) *image_int {
	/* check parameters */
	if xsize == 0 || ysize == 0 {
		fatal_error("new_image_int: invalid image size.")
	}

	/* get memory */
	image := image_int{xsize: xsize, ysize: ysize, data: make([]int, xsize*ysize)}

	return &image
}

/*----------------------------------------------------------------------------*/
/** Create a new image_int of size 'xsize' times 'ysize',
  initialized to the value 'fill_value'.
*/
func new_image_int_ini(xsize, ysize uint, fill_value int) *image_int {
	image := new_image_int(xsize, ysize) /* create image */

	/* initialize */
	for i := 0; i < len(image.data); i++ {
		image.data[i] = fill_value
	}

	return image
}

/*----------------------------------------------------------------------------*/
/** double image data type

  The pixel value at (x,y) is accessed by:

    image.data[ x + y * image.xsize ]

  with x and y integer.
*/
type image_double struct {
	xsize, ysize uint
	data         []float64
}

/*----------------------------------------------------------------------------*/
/** Create a new image_double of size 'xsize' times 'ysize'.
 */
func new_image_double(xsize, ysize uint) *image_double {
	/* check parameters */
	if xsize == 0 || ysize == 0 {
		fatal_error("new_image_double: invalid image size.")
	}

	/* get memory */
	image := image_double{xsize: xsize, ysize: ysize, data: make([]float64, xsize*ysize)}

	return &image
}

/*----------------------------------------------------------------------------*/
/** Create a new image_double of size 'xsize' times 'ysize'
  with the data pointed by 'data'.
*/
func new_image_double_ptr(xsize, ysize uint, data []float64) *image_double {
	/* check parameters */
	if xsize == 0 || ysize == 0 {
		fatal_error("new_image_double_ptr: invalid image size.")
	}

	if data == nil {
		fatal_error("new_image_double_ptr: nil data pointer.")
	}

	/* set image */
	image := image_double{xsize: xsize, ysize: ysize, data: data}

	return &image
}

/*----------------------------------------------------------------------------*/
/*----------------------------- Gaussian filter ------------------------------*/
/*----------------------------------------------------------------------------*/

/*----------------------------------------------------------------------------*/
/** Compute a Gaussian kernel of length 'kernel.dim',
  standard deviation 'sigma', and centered at value 'mean'.

  For example, if mean=0.5, the Gaussian will be centered
  in the middle point between values 'kernel.values[0]'
  and 'kernel.values[1]'.
*/
func gaussian_kernel(kernel *ntuple_list, sigma, mean float64) {
	var sum float64
	var val float64

	/* check parameters */
	if kernel == nil { // || kernel.values == nil {
		fatal_error("gaussian_kernel: invalid n-tuple 'kernel'. %v", kernel)
	}
	if sigma <= 0.0 {
		fatal_error("gaussian_kernel: 'sigma' must be positive.")
	}

	/* compute Gaussian kernel */
	kernel.size = 1

	for i := uint(0); i < kernel.dim; i++ {
		val = (float64(i) - mean) / sigma
		kernel.values = append(kernel.values, math.Exp(-0.5*val*val))
		sum += kernel.values[i]
	}

	/* normalization */
	if sum >= 0.0 {
		for i := uint(0); i < kernel.dim; i++ {
			kernel.values[i] /= sum
		}
	}
}

/*----------------------------------------------------------------------------*/
/** Scale the input image 'in' by a factor 'scale' by Gaussian sub-sampling.

  For example, scale=0.8 will give a result at 80% of the original size.

  The image is convolved with a Gaussian kernel
  @f[
      G(x,y) = \frac{1}{2\pi\sigma^2} e^{-\frac{x^2+y^2}{2\sigma^2}}
  @f]
  before the sub-sampling to prevent aliasing.

  The standard deviation sigma given by:
  -  sigma = sigma_scale / scale,   if scale <  1.0
  -  sigma = sigma_scale,           if scale >= 1.0

  To be able to sub-sample at non-integer steps, some interpolation
  is needed. In this implementation, the interpolation is done by
  the Gaussian kernel, so both operations (filtering and sampling)
  are done at the same time. The Gaussian kernel is computed
  centered on the coordinates of the required sample. In this way,
  when applied, it gives directly the result of convolving the image
  with the kernel and interpolated to that particular position.

  A fast algorithm is done using the separability of the Gaussian
  kernel. Applying the 2D Gaussian kernel is equivalent to applying
  first a horizontal 1D Gaussian kernel and then a vertical 1D
  Gaussian kernel (or the other way round). The reason is that
  @f[
      G(x,y) = G(x) * G(y)
  @f]
  where
  @f[
      G(x) = \frac{1}{\sqrt{2\pi}\sigma} e^{-\frac{x^2}{2\sigma^2}}.
  @f]
  The algorithm first applies a combined Gaussian kernel and sampling
  in the x axis, and then the combined Gaussian kernel and sampling
  in the y axis.
*/
func gaussian_sampler(in *image_double, scale, sigma_scale float64) *image_double {
	/* check parameters */
	if in == nil || in.data == nil || in.xsize == 0 || in.ysize == 0 {
		fatal_error("gaussian_sampler: invalid image.")
	}
	if scale <= 0.0 {
		fatal_error("gaussian_sampler: 'scale' must be positive.")
	}
	if sigma_scale <= 0.0 {
		fatal_error("gaussian_sampler: 'sigma_scale' must be positive.")
	}

	/* compute new image size and get memory for images */
	if float64(in.xsize)*scale > float64(math.MaxUint64) ||
		float64(in.ysize)*scale > float64(math.MaxUint64) {
		fatal_error("gaussian_sampler: the output image size exceeds the handled size.")
	}

	N := uint(math.Ceil(float64(in.xsize) * scale))
	M := uint(math.Ceil(float64(in.ysize) * scale))
	aux := new_image_double(N, in.ysize)
	out := new_image_double(N, M)

	/* sigma, kernel size and memory for the kernel */
	sigma := sigma_scale
	if scale < 1.0 {
		sigma = sigma_scale / scale
	}
	/*
	   The size of the kernel is selected to guarantee that the
	   the first discarded term is at least 10^prec times smaller
	   than the central value. For that, h should be larger than x, with
	     e^(-x^2/2sigma^2) = 1/10^prec.
	   Then,
	     x = sigma * math.Sqrt( 2 * prec * ln(10) ).
	*/
	prec := 3.0
	h := uint(math.Ceil(sigma * math.Sqrt(2.0*prec*math.Log(10.0))))
	n := 1 + 2*h /* kernel size */
	kernel := new_ntuple_list(n)

	/* auxiliary double image size variables */
	double_x_size := int(2 * in.xsize)
	double_y_size := int(2 * in.ysize)

	/* First subsampling: x axis */
	for x := uint(0); x < aux.xsize; x++ {
		/*
		   x   is the coordinate in the new image.
		   xx  is the corresponding x-value in the original size image.
		   xc  is the integer value, the pixel coordinate of xx.
		*/
		xx := float64(x) / scale
		/* coordinate (0.0,0.0) is in the center of pixel (0,0),
		   so the pixel with xc=0 get the values of xx from -0.5 to 0.5 */
		xc := int(math.Floor(xx + 0.5))
		gaussian_kernel(kernel, sigma, float64(h)+xx-float64(xc))
		/* the kernel must be computed for each x because the fine
		   offset xx-xc is different in each case */

		for y := uint(0); y < aux.ysize; y++ {
			sum := 0.0
			for i := uint(0); i < kernel.dim; i++ {
				j := xc - int(h) + int(i)

				/* symmetry boundary condition */
				for j < 0 {
					j += double_x_size
				}
				for j >= double_x_size {
					j -= double_x_size
				}
				if j >= int(in.xsize) {
					j = double_x_size - 1 - j
				}

				sum += in.data[uint(j)+y*in.xsize] * kernel.values[i]
			}
			aux.data[x+y*aux.xsize] = sum
		}
	}

	/* Second subsampling: y axis */
	for y := uint(0); y < out.ysize; y++ {
		/*
		   y   is the coordinate in the new image.
		   yy  is the corresponding x-value in the original size image.
		   yc  is the integer value, the pixel coordinate of xx.
		*/
		yy := float64(y) / scale
		/* coordinate (0.0,0.0) is in the center of pixel (0,0),
		   so the pixel with yc=0 get the values of yy from -0.5 to 0.5 */
		yc := int(math.Floor(yy + 0.5))
		gaussian_kernel(kernel, sigma, float64(h)+yy-float64(yc))
		/* the kernel must be computed for each y because the fine
		   offset yy-yc is different in each case */

		for x := uint(0); x < out.xsize; x++ {
			sum := 0.0
			for i := uint(0); i < kernel.dim; i++ {
				j := yc - int(h) + int(i)

				/* symmetry boundary condition */
				for j < 0 {
					j += double_y_size
				}
				for j >= double_y_size {
					j -= double_y_size
				}
				if j >= int(in.ysize) {
					j = double_y_size - 1 - j
				}

				sum += aux.data[x+uint(j)*aux.xsize] * kernel.values[i]
			}
			out.data[x+y*out.xsize] = sum
		}
	}

	return out
}

/*----------------------------------------------------------------------------*/
/*--------------------------------- Gradient ---------------------------------*/
/*----------------------------------------------------------------------------*/

/*----------------------------------------------------------------------------*/
/** Computes the direction of the level line of 'in' at each point.

  The result is:
  - an image_double with the angle at each pixel, or NOTDEF if not defined.
  - the image_double 'modgrad' (a pointer is passed as argument)
    with the gradient magnitude at each point.
  - a list of pixels 'list_p' roughly ordered by decreasing
    gradient magnitude. (The order is made by classifying points
    into bins by gradient magnitude. The parameters 'n_bins' and
    'max_grad' specify the number of bins and the gradient modulus
    at the highest bin. The pixels in the list would be in
    decreasing gradient magnitude, up to a precision of the size of
    the bins.)
*/
func ll_angle(in *image_double, threshold float64, list_p **coorlist,
	modgrad **image_double, n_bins uint) *image_double {
	var max_grad float64

	/* check parameters */
	if in == nil || in.data == nil || in.xsize == 0 || in.ysize == 0 {
		fatal_error("ll_angle: invalid image.")
	}
	if threshold < 0.0 {
		fatal_error("ll_angle: 'threshold' must be positive.")
	}
	if list_p == nil {
		fatal_error("ll_angle: nil pointer 'list_p'.")
	}
	if modgrad == nil {
		fatal_error("ll_angle: nil pointer 'modgrad'.")
	}
	if n_bins == 0 {
		fatal_error("ll_angle: 'n_bins' must be positive.")
	}

	/* image size shortcuts */
	n := in.ysize
	p := in.xsize

	/* allocate output image */
	g := new_image_double(in.xsize, in.ysize)

	/* get memory for the image of gradient modulus */
	*modgrad = new_image_double(in.xsize, in.ysize)

	/* get memory for "ordered" list of pixels */
	list := make([]coorlist, n*p)
	list_count := 0

	range_l_s := make([]*coorlist, n_bins)
	range_l_e := make([]*coorlist, n_bins)

	/* 'undefined' on the down and right boundaries */
	for x := uint(0); x < p; x++ {
		g.data[(n-1)*p+x] = NOTDEF
	}
	for y := uint(0); y < n; y++ {
		g.data[p*y+p-1] = NOTDEF
	}

	/* compute gradient on the remaining pixels */
	for x := uint(0); x < p-1; x++ {
		for y := uint(0); y < n-1; y++ {
			adr := y*p + x

			/*
			   Norm 2 computation using 2x2 pixel window:
			     A B
			     C D
			   and
			     com1 = D-A,  com2 = B-C.
			   Then
			     gx = B+D - (A+C)   horizontal difference
			     gy = C+D - (A+B)   vertical difference
			   com1 and com2 are just to avoid 2 additions.
			*/
			com1 := in.data[adr+p+1] - in.data[adr]
			com2 := in.data[adr+1] - in.data[adr+p]

			gx := com1 + com2 /* gradient x component */
			gy := com1 - com2 /* gradient y component */
			norm2 := gx*gx + gy*gy
			norm := math.Sqrt(norm2 / 4.0) /* gradient norm */

			(*modgrad).data[adr] = norm /* store gradient norm */

			if norm <= threshold { /* norm too small, gradient no defined */
				g.data[adr] = NOTDEF /* gradient angle not defined */
			} else {
				/* gradient angle computation */
				g.data[adr] = math.Atan2(gx, -gy)

				/* look for the maximum of the gradient */
				if norm > max_grad {
					max_grad = norm
				}
			}
		}
	}

	/* compute histogram of gradient values */
	for x := uint(0); x < p-1; x++ {
		for y := uint(0); y < n-1; y++ {
			norm := (*modgrad).data[y*p+x]

			/* store the point in the right bin according to its norm */
			i := uint(norm * float64(n_bins) / max_grad)
			if i >= n_bins {
				i = n_bins - 1
			}
			if range_l_e[i] == nil {
				range_l_s[i] = &list[list_count]
				range_l_e[i] = &list[list_count]
				list_count++
			} else {
				range_l_e[i].next = &list[list_count]
				range_l_e[i] = &list[list_count]
				list_count++
			}
			range_l_e[i].x = int(x)
			range_l_e[i].y = int(y)
			range_l_e[i].next = nil
		}
	}

	/* Make the list of pixels (almost) ordered by norm value.
	   It starts by the larger bin, so the list starts by the
	   pixels with the highest gradient value. Pixels would be ordered
	   by norm value, up to a precision given by max_grad/n_bins.
	*/
	i := n_bins - 1

	for i > 0 && range_l_s[i] == nil {
		i--
	}

	start := range_l_s[i]
	end := range_l_e[i]
	if start != nil {
		for i > 0 {
			i--
			if range_l_s[i] != nil {
				end.next = range_l_s[i]
				end = range_l_e[i]
			}
		}
	}
	*list_p = start

	return g
}

/*----------------------------------------------------------------------------*/
/** Is point with angle theta_test aligned to angle theta, up to precision 'prec'?
 */
func isaligned_(theta_test, theta, prec float64) bool {
	/* check parameters */
	if prec < 0.0 {
		fatal_error("isaligned: 'prec' must be positive.")
	}

	/* pixels whose level-line angle is not defined
	   are considered as NON-aligned */
	if theta_test == NOTDEF {
		return false /* there is no need to call the function
		   'double_equal' here because there is
		   no risk of problems related to the
		   comparison doubles, we are only
		   interested in the exact NOTDEF value */
	}

	/* it is assumed that 'theta' and 'a' are in the range [-pi,pi] */
	theta -= theta_test
	if theta < 0.0 {
		theta = -theta
	}
	if theta > M_3_2_PI {
		theta -= M_2__PI
		if theta < 0.0 {
			theta = -theta
		}
	}
	return theta <= prec
}

/** Is point (x,y) aligned to angle theta, up to precision 'prec'?
 */
func isaligned(x, y int, angles *image_double, theta, prec float64) bool {
	/* check parameters */
	if angles == nil || angles.data == nil {
		fatal_error("isaligned: invalid image 'angles'.")
	}
	if x < 0 || y < 0 || x >= int(angles.xsize) || y >= int(angles.ysize) {
		fatal_error("isaligned: (x,y) out of the image.")
	}
	if prec < 0.0 {
		fatal_error("isaligned: 'prec' must be positive.")
	}

	/* angle at pixel (x,y) */
	a := angles.data[uint(x)+uint(y)*angles.xsize]

	return isaligned_(a, theta, prec)
}

/*----------------------------------------------------------------------------*/
/** Absolute value angle difference.
 */
func angle_diff(a, b float64) float64 {
	a -= b
	for a <= -math.Pi {
		a += M_2__PI
	}
	for a > math.Pi {
		a -= M_2__PI
	}
	if a < 0.0 {
		a = -a
	}
	return a
}

/*----------------------------------------------------------------------------*/
/** Signed angle difference.
 */
func angle_diff_signed(a, b float64) float64 {
	a -= b
	for a <= -math.Pi {
		a += M_2__PI
	}
	for a > math.Pi {
		a -= M_2__PI
	}
	return a
}

/*----------------------------------------------------------------------------*/
/*----------------------------- NFA computation ------------------------------*/
/*----------------------------------------------------------------------------*/

/*----------------------------------------------------------------------------*/
/** Computes the natural logarithm of the absolute value of
  the gamma function of x using the Lanczos approximation.
  See http://www.rskey.org/gamma.htm

  The formula used is
  @f[
    \Gamma(x) = \frac{ \sum_{n=0}^{N} q_n x^n }{ \Pi_{n=0}^{N} (x+n) }
                (x+5.5)^{x+0.5} e^{-(x+5.5)}
  @f]
  so
  @f[
    \log\Gamma(x) = \log\left( \sum_{n=0}^{N} q_n x^n \right)
                    + (x+0.5) \math.Log(x+5.5) - (x+5.5) - \sum_{n=0}^{N} \math.Log(x+n)
  @f]
  and
    q0 = 75122.6331530,
    q1 = 80916.6278952,
    q2 = 36308.2951477,
    q3 = 8687.24529705,
    q4 = 1168.92649479,
    q5 = 83.8676043424,
    q6 = 2.50662827511.
*/
func log_gamma_lanczos(x float64) float64 {
	q := []float64{75122.6331530, 80916.6278952, 36308.2951477,
		8687.24529705, 1168.92649479, 83.8676043424,
		2.50662827511}
	a := (x+0.5)*math.Log(x+5.5) - (x + 5.5)
	b := 0.0

	for n := 0; n < len(q); n++ {
		a -= math.Log(x + float64(n))
		b += q[n] * math.Pow(x, float64(n))
	}
	return a + math.Log(b)
}

/*----------------------------------------------------------------------------*/
/** Computes the natural logarithm of the absolute value of
  the gamma function of x using Windschitl method.
  See http://www.rskey.org/gamma.htm

  The formula used is
  @f[
      \Gamma(x) = \sqrt{\frac{2\pi}{x}} \left( \frac{x}{e}
                  \sqrt{ x\sinh(1/x) + \frac{1}{810x^6} } \right)^x
  @f]
  so
  @f[
      \log\Gamma(x) = 0.5\math.Log(2\pi) + (x-0.5)\math.Log(x) - x
                    + 0.5x\log\left( x\sinh(1/x) + \frac{1}{810x^6} \right).
  @f]
  This formula is a good approximation when x > 15.
*/
func log_gamma_windschitl(x float64) float64 {
	return 0.918938533204673 + (x-0.5)*math.Log(x) - x + 0.5*x*math.Log(x*math.Sinh(1/x)+1/(810.0*math.Pow(x, 6.0)))
}

/*----------------------------------------------------------------------------*/
/** Computes the natural logarithm of the absolute value of
  the gamma function of x. When x>15 use log_gamma_windschitl(),
  otherwise use log_gamma_lanczos().
*/
func log_gamma(x float64) float64 {
	if x > 15.0 {
		return log_gamma_windschitl(x)
	} else {
		return log_gamma_lanczos(x)
	}
}

/*----------------------------------------------------------------------------*/
/** Size of the table to store already computed inverse values.
 */
var inv [100000]float64 /* table to keep computed inverse values */

/*----------------------------------------------------------------------------*/
/** Computes -log10(NFA).

  NFA stands for Number of False Alarms:
  @f[
      \mathrm{NFA} = NT \cdot B(n,k,p)
  @f]

  - NT       - number of tests
  - B(n,k,p) - tail of binomial distribution with parameters n,k and p:
  @f[
      B(n,k,p) = \sum_{j=k}^n
                 \left(\begin{array}{c}n\\j\end{array}\right)
                 p^{j} (1-p)^{n-j}
  @f]

  The value -log10(NFA) is equivalent but more intuitive than NFA:
  - -1 corresponds to 10 mean false alarms
  -  0 corresponds to 1 mean false alarm
  -  1 corresponds to 0.1 mean false alarms
  -  2 corresponds to 0.01 mean false alarms
  -  ...

  Used this way, the bigger the value, better the detection,
  and a logarithmic scale is used.

  @param n,k,p binomial parameters.
  @param logNT logarithm of Number of Tests

  The computation is based in the gamma function by the following
  relation:
  @f[
      \left(\begin{array}{c}n\\k\end{array}\right)
      = \frac{ \Gamma(n+1) }{ \Gamma(k+1) \cdot \Gamma(n-k+1) }.
  @f]
  We use efficient algorithms to compute the logarithm of
  the gamma function.

  To make the computation faster, not all the sum is computed, part
  of the terms are neglected based on a bound to the error obtained
  (an error of 10% in the result is accepted).
*/
func nfa(n, k int, p, logNT float64) float64 {
	tolerance := 0.1 /* an error of 10% in the result is accepted */

	/* check parameters */
	if n < 0 || k < 0 || k > n || p <= 0.0 || p >= 1.0 {
		fatal_error("nfa: wrong n, k or p values.")
	}

	/* trivial cases */
	if n == 0 || k == 0 {
		return -logNT
	}
	if n == k {
		return -logNT - float64(n)*math.Log10(p)
	}

	/* probability term */
	p_term := p / (1.0 - p)

	/* compute the first term of the series */
	/*
	   binomial_tail(n,k,p) = sum_{i=k}^n bincoef(n,i) * p^i * (1-p)^{n-i}
	   where bincoef(n,i) are the binomial coefficients.
	   But
	     bincoef(n,k) = gamma(n+1) / ( gamma(k+1) * gamma(n-k+1) ).
	   We use this to compute the first term. Actually the log of it.
	*/
	log1term := log_gamma(float64(n)+1.0) - log_gamma(float64(k)+1.0) -
		log_gamma(float64(n-k)+1.0) + float64(k)*math.Log(p) + float64(n-k)*math.Log(1.0-p)
	term := math.Exp(log1term)

	/* in some cases no more computations are needed */
	if double_equal(term, 0.0) { /* the first term is almost zero */
		if float64(k) > float64(n)*p { /* at begin or end of the tail?  */
			return -log1term/math.Ln10 - logNT /* end: use just the first term  */
		} else {
			return -logNT /* begin: the tail is roughly 1  */
		}
	}

	/* compute more terms if needed */
	bin_tail := term
	for i := k + 1; i <= n; i++ {
		/*
		   As
		     term_i = bincoef(n,i) * p^i * (1-p)^(n-i)
		   and
		     bincoef(n,i)/bincoef(n,i-1) = n-1+1 / i,
		   then,
		     term_i / term_i-1 = (n-i+1)/i * p/(1-p)
		   and
		     term_i = term_i-1 * (n-i+1)/i * p/(1-p).
		   1/i is stored in a table as they are computed,
		   because divisions are expensive.
		   p/(1-p) is computed only once and stored in 'p_term'.
		*/
		var v float64
		if i < len(inv) {
			if inv[i] == 0.0 {
				inv[i] = 1.0 / float64(i)
			}

			v = inv[i]
		} else {
			v = 1.0 / float64(i)
		}
		bin_term := float64(n-i+1) * v

		mult_term := bin_term * p_term
		term *= mult_term
		bin_tail += term
		if bin_term < 1.0 {
			/* When bin_term<1 then mult_term_j<mult_term_i for j>i.
			   Then, the error on the binomial tail when truncated at
			   the i term can be bounded by a geometric series of form
			   term_i * sum mult_term_i^j.                            */
			err := term * ((1.0-math.Pow(mult_term, float64(n-i+1)))/
				(1.0-mult_term) - 1.0)

			/* One wants an error at most of tolerance*final_result, or:
			   tolerance * math.Abs(-log10(bin_tail)-logNT).
			   Now, the error that can be accepted on bin_tail is
			   given by tolerance*final_result divided by the derivative
			   of -log10(x) when x=bin_tail. that is:
			   tolerance * math.Abs(-log10(bin_tail)-logNT) / (1/bin_tail)
			   Finally, we truncate the tail if the error is less than:
			   tolerance * math.Abs(-log10(bin_tail)-logNT) * bin_tail        */
			if err < tolerance*math.Abs(-math.Log10(bin_tail)-logNT)*bin_tail {
				break
			}
		}
	}
	return -math.Log10(bin_tail) - logNT
}

/*----------------------------------------------------------------------------*/
/*--------------------------- Rectangle structure ----------------------------*/
/*----------------------------------------------------------------------------*/

/*----------------------------------------------------------------------------*/
/** Rectangle structure: line segment with width.
 */
type rect struct {
	x1, y1, x2, y2 float64 /* first and second point of the line segment */
	width          float64 /* rectangle width */
	x, y           float64 /* center of the rectangle */
	theta          float64 /* angle */
	dx, dy         float64 /* (dx,dy) is vector oriented as the line segment */
	prec           float64 /* tolerance angle */
	p              float64 /* probability of a point with angle within 'prec' */
	length         float64
}

/*----------------------------------------------------------------------------*/
/** Copy one rectangle structure to another.
 */
func rect_copy(in, out *rect) {
	/* check parameters */
	if in == nil || out == nil {
		fatal_error("rect_copy: invalid 'in' or 'out'.")
	}

	/* copy values */
	out.x1 = in.x1
	out.y1 = in.y1
	out.x2 = in.x2
	out.y2 = in.y2
	out.width = in.width
	out.x = in.x
	out.y = in.y
	out.theta = in.theta
	out.dx = in.dx
	out.dy = in.dy
	out.prec = in.prec
	out.p = in.p
}

/*----------------------------------------------------------------------------*/
/** Rectangle points iterator.

  The integer coordinates of pixels inside a rectangle are
  iteratively explored. This structure keep track of the process and
  functions ri_ini(), ri_inc(), ri_end(), and ri_del() are used in
  the process. An example of how to use the iterator is as follows:
  \code

    struct rect * rec = XXX; // some rectangle
    rect_iter * i
    for( i=ri_ini(rec); !ri_end(i); ri_inc(i) )
      {
        // your code, using 'i.x' and 'i.y' as coordinates
      }
    ri_del(i); // delete iterator

  \endcode
  The pixels are explored 'column' by 'column', where we call
  'column' a set of pixels with the same x value that are inside the
  rectangle. The following is an schematic representation of a
  rectangle, the 'column' being explored is marked by colons, and
  the current pixel being explored is 'x,y'.
  \verbatim

            vx[1],vy[1]
               *   *
              *       *
             *           *
            *               ye
           *                :  *
      vx[0],vy[0]           :     *
             *              :        *
                *          x,y          *
                   *        :              *
                      *     :            vx[2],vy[2]
                         *  :                *
      y                     ys              *
      ^                        *           *
      |                           *       *
      |                              *   *
      +--. x                      vx[3],vy[3]

  \endverbatim
  The first 'column' to be explored is the one with the smaller x
  value. Each 'column' is explored starting from the pixel of the
  'column' (inside the rectangle) with the smallest y value.

  The four corners of the rectangle are stored in order that rotates
  around the corners at the arrays 'vx[]' and 'vy[]'. The first
  point is always the one with smaller x value.

  'x' and 'y' are the coordinates of the pixel being explored. 'ys'
  and 'ye' are the start and end values of the current column being
  explored. So, 'ys' < 'ye'.
*/
type rect_iter struct {
	vx     [4]float64 /* rectangle's corner X coordinates in circular order */
	vy     [4]float64 /* rectangle's corner Y coordinates in circular order */
	ys, ye float64    /* start and end Y values of current 'column' */
	x, y   int        /* coordinates of currently explored pixel */
}

/*----------------------------------------------------------------------------*/
/** Interpolate y value corresponding to 'x' value given, in
  the line 'x1,y1' to 'x2,y2'; if 'x1=x2' return the smaller
  of 'y1' and 'y2'.

  The following restrictions are required:
  - x1 <= x2
  - x1 <= x
  - x  <= x2
*/
func inter_low(x, x1, y1, x2, y2 float64) float64 {
	/* check parameters */
	if x1 > x2 || x < x1 || x > x2 {
		fatal_error("inter_low: unsuitable input, 'x1>x2' or 'x<x1' or 'x>x2'.")
	}

	/* interpolation */
	if double_equal(x1, x2) && y1 < y2 {
		return y1
	}
	if double_equal(x1, x2) && y1 > y2 {
		return y2
	}
	return y1 + (x-x1)*(y2-y1)/(x2-x1)
}

/*----------------------------------------------------------------------------*/
/** Interpolate y value corresponding to 'x' value given, in
  the line 'x1,y1' to 'x2,y2'; if 'x1=x2' return the larger
  of 'y1' and 'y2'.

  The following restrictions are required:
  - x1 <= x2
  - x1 <= x
  - x  <= x2
*/
func inter_hi(x, x1, y1, x2, y2 float64) float64 {
	/* check parameters */
	if x1 > x2 || x < x1 || x > x2 {
		fatal_error("inter_hi: unsuitable input, 'x1>x2' or 'x<x1' or 'x>x2'.")
	}

	/* interpolation */
	if double_equal(x1, x2) && y1 < y2 {
		return y2
	}
	if double_equal(x1, x2) && y1 > y2 {
		return y1
	}
	return y1 + (x-x1)*(y2-y1)/(x2-x1)
}

/*----------------------------------------------------------------------------*/
/** Check if the iterator finished the full iteration.

  See details in \ref rect_iter
*/
func ri_end(i *rect_iter) bool {
	/* check input */
	if i == nil {
		fatal_error("ri_end: nil iterator.")
	}

	/* if the current x value is larger than the largest
	   x value in the rectangle (vx[2]), we know the full
	   exploration of the rectangle is finished. */
	return float64(i.x) > i.vx[2]
}

/*----------------------------------------------------------------------------*/
/** Increment a rectangle iterator.

  See details in \ref rect_iter
*/
func ri_inc(i *rect_iter) {
	/* check input */
	if i == nil {
		fatal_error("ri_inc: nil iterator.")
	}

	/* if not at end of exploration,
	   increase y value for next pixel in the 'column' */
	if !ri_end(i) {
		i.y++
	}

	/* if the end of the current 'column' is reached,
	   and it is not the end of exploration,
	   advance to the next 'column' */
	for float64(i.y) > i.ye && !ri_end(i) {
		/* increase x, next 'column' */
		i.x++

		/* if end of exploration, return */
		if ri_end(i) {
			return
		}

		/* update lower y limit (start) for the new 'column'.

		   We need to interpolate the y value that corresponds to the
		   lower side of the rectangle. The first thing is to decide if
		   the corresponding side is

		     vx[0],vy[0] to vx[3],vy[3] or
		     vx[3],vy[3] to vx[2],vy[2]

		   Then, the side is interpolated for the x value of the
		   'column'. But, if the side is vertical (as it could happen if
		   the rectangle is vertical and we are dealing with the first
		   or last 'columns') then we pick the lower value of the side
		   by using 'inter_low'.
		*/
		if float64(i.x) < i.vx[3] {
			i.ys = inter_low(float64(i.x), i.vx[0], i.vy[0], i.vx[3], i.vy[3])
		} else {
			i.ys = inter_low(float64(i.x), i.vx[3], i.vy[3], i.vx[2], i.vy[2])
		}

		/* update upper y limit (end) for the new 'column'.

		   We need to interpolate the y value that corresponds to the
		   upper side of the rectangle. The first thing is to decide if
		   the corresponding side is

		     vx[0],vy[0] to vx[1],vy[1] or
		     vx[1],vy[1] to vx[2],vy[2]

		   Then, the side is interpolated for the x value of the
		   'column'. But, if the side is vertical (as it could happen if
		   the rectangle is vertical and we are dealing with the first
		   or last 'columns') then we pick the lower value of the side
		   by using 'inter_low'.
		*/
		if float64(i.x) < i.vx[1] {
			i.ye = inter_hi(float64(i.x), i.vx[0], i.vy[0], i.vx[1], i.vy[1])
		} else {
			i.ye = inter_hi(float64(i.x), i.vx[1], i.vy[1], i.vx[2], i.vy[2])
		}

		/* new y */
		i.y = int(math.Ceil(i.ys))
	}
}

/*----------------------------------------------------------------------------*/
/** Create and initialize a rectangle iterator.

  See details in \ref rect_iter
*/
func ri_ini(r *rect) *rect_iter {
	var vx [4]float64
	var vy [4]float64
	var i rect_iter
	var offset int

	/* check parameters */
	if r == nil {
		fatal_error("ri_ini: invalid rectangle.")
	}

	/* build list of rectangle corners ordered
	   in a circular way around the rectangle */
	vx[0] = r.x1 - r.dy*r.width/2.0
	vy[0] = r.y1 + r.dx*r.width/2.0
	vx[1] = r.x2 - r.dy*r.width/2.0
	vy[1] = r.y2 + r.dx*r.width/2.0
	vx[2] = r.x2 + r.dy*r.width/2.0
	vy[2] = r.y2 - r.dx*r.width/2.0
	vx[3] = r.x1 + r.dy*r.width/2.0
	vy[3] = r.y1 - r.dx*r.width/2.0

	/* compute rotation of index of corners needed so that the first
	   point has the smaller x.

	   if one side is vertical, thus two corners have the same smaller x
	   value, the one with the largest y value is selected as the first.
	*/
	if r.x1 < r.x2 && r.y1 <= r.y2 {
		offset = 0
	} else if r.x1 >= r.x2 && r.y1 < r.y2 {
		offset = 1
	} else if r.x1 > r.x2 && r.y1 >= r.y2 {
		offset = 2
	} else {
		offset = 3
	}

	/* apply rotation of index. */
	for n := 0; n < 4; n++ {
		i.vx[n] = vx[(offset+n)%4]
		i.vy[n] = vy[(offset+n)%4]
	}

	/* Set an initial condition.

	   The values are set to values that will cause 'ri_inc' (that will
	   be called immediately) to initialize correctly the first 'column'
	   and compute the limits 'ys' and 'ye'.

	   'y' is set to the integer value of vy[0], the starting corner.

	   'ys' and 'ye' are set to very small values, so 'ri_inc' will
	   notice that it needs to start a new 'column'.

	   The smallest integer coordinate inside of the rectangle is
	   'math.Ceil(vx[0])'. The current 'x' value is set to that value minus
	   one, so 'ri_inc' (that will increase x by one) will advance to
	   the first 'column'.
	*/
	i.x = int(math.Ceil(i.vx[0])) - 1
	i.y = int(math.Ceil(i.vy[0]))
	i.ys = -DBL_MAX
	i.ye = -DBL_MAX

	/* advance to the first pixel */
	ri_inc(&i)

	return &i
}

/*----------------------------------------------------------------------------*/
/** Compute a rectangle's NFA value.
 */
func rect_nfa(rec *rect, angles *image_double, logNT float64) float64 {
	pts := 0
	alg := 0

	/* check parameters */
	if rec == nil {
		fatal_error("rect_nfa: invalid rectangle.")
	}
	if angles == nil {
		fatal_error("rect_nfa: invalid 'angles'.")
	}

	/* compute the total number of pixels and of aligned points in 'rec' */
	for i := ri_ini(rec); !ri_end(i); ri_inc(i) { /* rectangle iterator */
		if i.x >= 0 && i.y >= 0 &&
			i.x < int(angles.xsize) && i.y < int(angles.ysize) {
			pts++ /* total number of pixels counter */
			if isaligned(i.x, i.y, angles, rec.theta, rec.prec) {
				alg++ /* aligned points counter */
			}
		}
	}

	return nfa(pts, alg, rec.p, logNT) /* compute NFA value */
}

/*----------------------------------------------------------------------------*/
/*---------------------------------- Regions ---------------------------------*/
/*----------------------------------------------------------------------------*/

/*----------------------------------------------------------------------------*/
/** Compute region's angle as the principal inertia axis of the region.

  The following is the region inertia matrix A:
  @f[

      A = \left(\begin{array}{cc}
                                  Ixx & Ixy \\
                                  Ixy & Iyy \\
           \end{array}\right)

  @f]
  where

    Ixx =   sum_i G(i).(y_i - cx)^2

    Iyy =   sum_i G(i).(x_i - cy)^2

    Ixy = - sum_i G(i).(x_i - cx).(y_i - cy)

  and
  - G(i) is the gradient norm at pixel i, used as pixel's weight.
  - x_i and y_i are the coordinates of pixel i.
  - cx and cy are the coordinates of the center of th region.

  lambda1 and lambda2 are the eigenvalues of matrix A,
  with lambda1 >= lambda2. They are found by solving the
  characteristic polynomial:

    det( lambda I - A) = 0

  that gives:

    lambda1 = ( Ixx + Iyy + math.Sqrt( (Ixx-Iyy)^2 + 4.0*Ixy*Ixy) ) / 2

    lambda2 = ( Ixx + Iyy - math.Sqrt( (Ixx-Iyy)^2 + 4.0*Ixy*Ixy) ) / 2

  To get the line segment direction we want to get the angle the
  eigenvector associated to the smallest eigenvalue. We have
  to solve for a,b in:

    a.Ixx + b.Ixy = a.lambda2

    a.Ixy + b.Iyy = b.lambda2

  We want the angle theta = atan(b/a). It can be computed with
  any of the two equations:

    theta = atan( (lambda2-Ixx) / Ixy )

  or

    theta = atan( Ixy / (lambda2-Iyy) )

  When |Ixx| > |Iyy| we use the first, otherwise the second (just to
  get better numeric precision).
*/
func get_theta(reg []point, reg_size int, x, y float64,
	modgrad *image_double, reg_angle, prec float64) float64 {
	Ixx := 0.0
	Iyy := 0.0
	Ixy := 0.0

	/* check parameters */
	if reg == nil {
		fatal_error("get_theta: invalid region.")
	}
	if reg_size <= 1 {
		fatal_error("get_theta: region size <= 1.")
	}
	if modgrad == nil || modgrad.data == nil {
		fatal_error("get_theta: invalid 'modgrad'.")
	}
	if prec < 0.0 {
		fatal_error("get_theta: 'prec' must be positive.")
	}

	/* compute inertia matrix */
	for i := 0; i < reg_size; i++ {
		weight := modgrad.data[reg[i].x+reg[i].y*int(modgrad.xsize)]
		Ixx += (float64(reg[i].y) - y) * (float64(reg[i].y) - y) * weight
		Iyy += (float64(reg[i].x) - x) * (float64(reg[i].x) - x) * weight
		Ixy -= (float64(reg[i].x) - x) * (float64(reg[i].y) - y) * weight
	}
	if double_equal(Ixx, 0.0) && double_equal(Iyy, 0.0) && double_equal(Ixy, 0.0) {
		fatal_error("get_theta: null inertia matrix.")
	}

	/* compute smallest eigenvalue */
	lambda := 0.5 * (Ixx + Iyy - math.Sqrt((Ixx-Iyy)*(Ixx-Iyy)+4.0*Ixy*Ixy))

	/* compute angle */
	var theta float64
	if math.Abs(Ixx) > math.Abs(Iyy) {
		theta = math.Atan2(lambda-Ixx, Ixy)
	} else {
		theta = math.Atan2(Ixy, lambda-Iyy)
	}

	/* The previous procedure doesn't cares about orientation,
	   so it could be wrong by 180 degrees. Here is corrected if necessary. */
	if angle_diff(theta, reg_angle) > prec {
		theta += math.Pi
	}

	return theta
}

func get_reg_union_theta(reg []int, reg_size int, x, y float64,
	points []*rect, reg_angle, prec float64) float64 {
	Ixx := 0.0
	Iyy := 0.0
	Ixy := 0.0

	if reg_size == 1 {
		return points[reg[0]].theta
	}

	/* check parameters */
	if reg == nil {
		fatal_error("get_reg_union_theta: invalid region.")
	}
	if prec < 0.0 {
		fatal_error("get_reg_union_theta: 'prec' must be positive.")
	}

	/* compute inertia matrix */
	for i := 0; i < reg_size; i++ {
		weight := points[reg[i]].length
		Ixx += float64(points[reg[i]].y-y) * float64(points[reg[i]].y-y) * weight
		Iyy += float64(points[reg[i]].x-x) * float64(points[reg[i]].x-x) * weight
		Ixy -= float64(points[reg[i]].x-x) * float64(points[reg[i]].y-y) * weight
	}
	if double_equal(Ixx, 0.0) && double_equal(Iyy, 0.0) && double_equal(Ixy, 0.0) {
		fatal_error("get_reg_union_theta: null inertia matrix.")
	}

	/* compute smallest eigenvalue */
	lambda := 0.5 * (Ixx + Iyy - math.Sqrt((Ixx-Iyy)*(Ixx-Iyy)+4.0*Ixy*Ixy))

	/* compute angle */
	var theta float64
	if math.Abs(Ixx) > math.Abs(Iyy) {
		theta = math.Atan2(lambda-Ixx, Ixy)
	} else {
		theta = math.Atan2(Ixy, lambda-Iyy)
	}

	/* The previous procedure doesn't cares about orientation,
	   so it could be wrong by 180 degrees. Here is corrected if necessary. */
	if angle_diff(theta, reg_angle) > prec {
		theta += math.Pi
	}

	return theta
}

/*----------------------------------------------------------------------------*/
/** Computes a rectangle that covers a region of points.
 */
func region2rect(reg []point, reg_size int,
	modgrad *image_double, reg_angle, prec, p float64,
	rec *rect) {
	/* check parameters */
	if reg == nil {
		fatal_error("region2rect: invalid region.")
	}
	if reg_size <= 1 {
		fatal_error("region2rect: region size <= 1.")
	}
	if modgrad == nil || modgrad.data == nil {
		fatal_error("region2rect: invalid image 'modgrad'.")
	}
	if rec == nil {
		fatal_error("region2rect: invalid 'rec'.")
	}

	/* center of the region:

	   It is computed as the weighted sum of the coordinates
	   of all the pixels in the region. The norm of the gradient
	   is used as the weight of a pixel. The sum is as follows:
	     cx = \sum_i G(i).x_i
	     cy = \sum_i G(i).y_i
	   where G(i) is the norm of the gradient of pixel i
	   and x_i,y_i are its coordinates.
	*/
	x := 0.0
	y := 0.0
	sum := 0.0

	for i := 0; i < reg_size; i++ {
		weight := modgrad.data[reg[i].x+reg[i].y*int(modgrad.xsize)]
		x += float64(reg[i].x) * weight
		y += float64(reg[i].y) * weight
		sum += weight
	}
	if sum <= 0.0 {
		fatal_error("region2rect: weights sum equal to zero.")
	}
	x /= sum
	y /= sum

	/* theta */
	theta := get_theta(reg, reg_size, x, y, modgrad, reg_angle, prec)

	/* length and width:

	   'l' and 'w' are computed as the distance from the center of the
	   region to pixel i, projected along the rectangle axis (dx,dy) and
	   to the orthogonal axis (-dy,dx), respectively.

	   The length of the rectangle goes from l_min to l_max, where l_min
	   and l_max are the minimum and maximum values of l in the region.
	   Analogously, the width is selected from w_min to w_max, where
	   w_min and w_max are the minimum and maximum of w for the pixels
	   in the region.
	*/
	dx := math.Cos(theta)
	dy := math.Sin(theta)
	l_min := 0.0
	l_max := 0.0
	w_min := 0.0
	w_max := 0.0
	for i := 0; i < reg_size; i++ {
		l := (float64(reg[i].x)-x)*dx + (float64(reg[i].y)-y)*dy
		w := -(float64(reg[i].x)-x)*dy + (float64(reg[i].y)-y)*dx

		if l > l_max {
			l_max = l
		}
		if l < l_min {
			l_min = l
		}
		if w > w_max {
			w_max = w
		}
		if w < w_min {
			w_min = w
		}
	}

	/* store values */
	rec.x1 = x + l_min*dx
	rec.y1 = y + l_min*dy
	rec.x2 = x + l_max*dx
	rec.y2 = y + l_max*dy
	rec.width = w_max - w_min
	rec.x = x
	rec.y = y
	rec.theta = theta
	rec.dx = dx
	rec.dy = dy
	rec.prec = prec
	rec.p = p

	/* we impose a minimal width of one pixel

	   A sharp horizontal or vertical step would produce a perfectly
	   horizontal or vertical region. The width computed would be
	   zero. But that corresponds to a one pixels width transition in
	   the image.
	*/
	if rec.width < 1.0 {
		rec.width = 1.0
	}
}

/*----------------------------------------------------------------------------*/
/** Build a region of pixels that share the same angle, up to a
  tolerance 'prec', starting at point (x,y).
*/
func region_grow(x, y int, angles *image_double, reg []point,
	reg_size *int, reg_angle *float64, used *image_bool,
	prec float64) {
	/* check parameters */
	if x < 0 || y < 0 || x >= int(angles.xsize) || y >= int(angles.ysize) {
		fatal_error("region_grow: (x,y) out of the image.")
	}
	if angles == nil || angles.data == nil {
		fatal_error("region_grow: invalid image 'angles'.")
	}
	if reg == nil {
		fatal_error("region_grow: invalid 'reg'.")
	}
	if reg_size == nil {
		fatal_error("region_grow: invalid pointer 'reg_size'.")
	}
	if reg_angle == nil {
		fatal_error("region_grow: invalid pointer 'reg_angle'.")
	}
	if used == nil || used.data == nil {
		fatal_error("region_grow: invalid image 'used'.")
	}

	/* first point of the region */
	*reg_size = 1
	reg[0].x = x
	reg[0].y = y
	*reg_angle = angles.data[x+y*int(angles.xsize)] /* region's angle */
	sumdx := math.Cos(*reg_angle)
	sumdy := math.Sin(*reg_angle)
	used.data[x+y*int(used.xsize)] = true

	/* try neighbors as new region points */
	for i := 0; i < *reg_size; i++ {
		for xx := reg[i].x - 1; xx <= reg[i].x+1; xx++ {
			for yy := reg[i].y - 1; yy <= reg[i].y+1; yy++ {
				if xx >= 0 && yy >= 0 && xx < int(used.xsize) && yy < int(used.ysize) &&
					used.data[xx+yy*int(used.xsize)] != true &&
					isaligned(xx, yy, angles, *reg_angle, prec) {
					/* add point */
					used.data[xx+yy*int(used.xsize)] = true
					reg[*reg_size].x = xx
					reg[*reg_size].y = yy
					(*reg_size)++

					/* update region's angle */
					sumdx += math.Cos(angles.data[xx+yy*int(angles.xsize)])
					sumdy += math.Sin(angles.data[xx+yy*int(angles.xsize)])
					*reg_angle = math.Atan2(sumdy, sumdx)
				}
			}
		}
	}
}

/*----------------------------------------------------------------------------*/
/** Try some rectangles variations to improve NFA value. Only if the
  rectangle is not meaningful (i.e., log_nfa <= log_eps).
*/
func rect_improve(rec *rect, angles *image_double,
	logNT, log_eps float64) float64 {
	log_nfa := rect_nfa(rec, angles, logNT)
	var log_nfa_new float64
	delta := 0.5
	delta_2 := delta / 2.0

	if log_nfa > log_eps {
		return log_nfa
	}

	var r rect

	/* try finer precisions */
	rect_copy(rec, &r)
	for n := 0; n < 5; n++ {
		r.p /= 2.0
		r.prec = r.p * math.Pi
		log_nfa_new = rect_nfa(&r, angles, logNT)
		if log_nfa_new > log_nfa {
			log_nfa = log_nfa_new
			rect_copy(&r, rec)
		}
	}

	if log_nfa > log_eps {
		return log_nfa
	}

	/* try to reduce width */
	rect_copy(rec, &r)
	for n := 0; n < 5; n++ {
		if (r.width - delta) >= 0.5 {
			r.width -= delta
			log_nfa_new = rect_nfa(&r, angles, logNT)
			if log_nfa_new > log_nfa {
				rect_copy(&r, rec)
				log_nfa = log_nfa_new
			}
		}
	}

	if log_nfa > log_eps {
		return log_nfa
	}

	/* try to reduce one side of the rectangle */
	rect_copy(rec, &r)
	for n := 0; n < 5; n++ {
		if (r.width - delta) >= 0.5 {
			r.x1 += -r.dy * delta_2
			r.y1 += r.dx * delta_2
			r.x2 += -r.dy * delta_2
			r.y2 += r.dx * delta_2
			r.width -= delta
			log_nfa_new = rect_nfa(&r, angles, logNT)
			if log_nfa_new > log_nfa {
				rect_copy(&r, rec)
				log_nfa = log_nfa_new
			}
		}
	}

	if log_nfa > log_eps {
		return log_nfa
	}

	/* try to reduce the other side of the rectangle */
	rect_copy(rec, &r)
	for n := 0; n < 5; n++ {
		if (r.width - delta) >= 0.5 {
			r.x1 -= -r.dy * delta_2
			r.y1 -= r.dx * delta_2
			r.x2 -= -r.dy * delta_2
			r.y2 -= r.dx * delta_2
			r.width -= delta
			log_nfa_new = rect_nfa(&r, angles, logNT)
			if log_nfa_new > log_nfa {
				rect_copy(&r, rec)
				log_nfa = log_nfa_new
			}
		}
	}

	if log_nfa > log_eps {
		return log_nfa
	}

	/* try even finer precisions */
	rect_copy(rec, &r)
	for n := 0; n < 5; n++ {
		r.p /= 2.0
		r.prec = r.p * math.Pi
		log_nfa_new = rect_nfa(&r, angles, logNT)
		if log_nfa_new > log_nfa {
			log_nfa = log_nfa_new
			rect_copy(&r, rec)
		}
	}

	return log_nfa
}

/*----------------------------------------------------------------------------*/
/** Reduce the region size, by elimination the points far from the
  starting point, until that leads to rectangle with the right
  density of region points or to discard the region if too small.
*/
func reduce_region_radius(reg []point, reg_size *int,
	modgrad *image_double, reg_angle, prec, p float64,
	rec *rect,
	used *image_bool, angles *image_double,
	density_th float64) bool {
	/* check parameters */
	if reg == nil {
		fatal_error("reduce_region_radius: invalid pointer 'reg'.")
	}
	if reg_size == nil {
		fatal_error("reduce_region_radius: invalid pointer 'reg_size'.")
	}
	if prec < 0.0 {
		fatal_error("reduce_region_radius: 'prec' must be positive.")
	}
	if rec == nil {
		fatal_error("reduce_region_radius: invalid pointer 'rec'.")
	}
	if used == nil || used.data == nil {
		fatal_error("reduce_region_radius: invalid image 'used'.")
	}
	if angles == nil || angles.data == nil {
		fatal_error("reduce_region_radius: invalid image 'angles'.")
	}

	/* compute region points density */
	density := float64(*reg_size) / (dist(rec.x1, rec.y1, rec.x2, rec.y2) * rec.width)

	/* if the density criterion is satisfied there is nothing to do */
	if density >= density_th {
		return true
	}

	/* compute region's radius */
	xc := float64(reg[0].x)
	yc := float64(reg[0].y)
	rad1 := dist(xc, yc, rec.x1, rec.y1)
	rad2 := dist(xc, yc, rec.x2, rec.y2)
	rad := rad2
	if rad1 > rad2 {
		rad = rad1
	}

	/* while the density criterion is not satisfied, remove farther pixels */
	for density < density_th {
		rad *= 0.75 /* reduce region's radius to 75% of its value */

		/* remove points from the region and update 'used' map */
		for i := 0; i < *reg_size; i++ {
			if dist(xc, yc, float64(reg[i].x), float64(reg[i].y)) > rad {
				/* point not kept, mark it as false */
				used.data[reg[i].x+reg[i].y*int(used.xsize)] = false
				/* remove point from the region */
				reg[i].x = reg[*reg_size-1].x /* if i==*reg_size-1 copy itself */
				reg[i].y = reg[*reg_size-1].y
				(*reg_size)--
				i-- /* to avoid skipping one point */
			}
		}

		/* reject if the region is too small.
		   2 is the minimal region size for 'region2rect' to work. */
		if *reg_size < 2 {
			return false
		}

		/* re-compute rectangle */
		region2rect(reg, *reg_size, modgrad, reg_angle, prec, p, rec)

		/* re-compute region points density */
		density = float64(*reg_size) / (dist(rec.x1, rec.y1, rec.x2, rec.y2) * rec.width)
	}

	/* if this point is reached, the density criterion is satisfied */
	return true
}

/*----------------------------------------------------------------------------*/
/** Refine a rectangle.

  For that, an estimation of the angle tolerance is performed by the
  standard deviation of the angle at points near the region's
  starting point. Then, a new region is grown starting from the same
  point, but using the estimated angle tolerance. If this fails to
  produce a rectangle with the right density of region points,
  'reduce_region_radius' is called to try to satisfy this condition.
*/
func refine(reg []point, reg_size *int, modgrad *image_double,
	reg_angle, prec, p float64, rec *rect,
	used *image_bool, angles *image_double, density_th float64) bool {

	/* check parameters */
	if reg == nil {
		fatal_error("refine: invalid pointer 'reg'.")
	}
	if reg_size == nil {
		fatal_error("refine: invalid pointer 'reg_size'.")
	}
	if prec < 0.0 {
		fatal_error("refine: 'prec' must be positive.")
	}
	if rec == nil {
		fatal_error("refine: invalid pointer 'rec'.")
	}
	if used == nil || used.data == nil {
		fatal_error("refine: invalid image 'used'.")
	}
	if angles == nil || angles.data == nil {
		fatal_error("refine: invalid image 'angles'.")
	}

	/* compute region points density */
	density := float64(*reg_size) /
		(dist(rec.x1, rec.y1, rec.x2, rec.y2) * rec.width)

	/* if the density criterion is satisfied there is nothing to do */
	if density >= density_th {
		return true
	}

	/*------ First try: reduce angle tolerance ------*/

	/* compute the new mean angle and tolerance */
	xc := float64(reg[0].x)
	yc := float64(reg[0].y)
	ang_c := angles.data[reg[0].x+reg[0].y*int(angles.xsize)]
	sum := 0.0
	s_sum := 0.0
	n := 0
	for i := 0; i < *reg_size; i++ {
		used.data[reg[i].x+reg[i].y*int(used.xsize)] = false
		if dist(xc, yc, float64(reg[i].x), float64(reg[i].y)) < rec.width {
			angle := angles.data[reg[i].x+reg[i].y*int(angles.xsize)]
			ang_d := angle_diff_signed(angle, ang_c)
			sum += ang_d
			s_sum += ang_d * ang_d
			n++
		}
	}
	mean_angle := sum / float64(n)
	tau := 2.0 * math.Sqrt((s_sum-2.0*mean_angle*sum)/float64(n)+
		mean_angle*mean_angle) /* 2 * standard deviation */

	/* find a new region from the same starting point and new angle tolerance */
	region_grow(reg[0].x, reg[0].y, angles, reg, reg_size, &reg_angle, used, tau)

	/* if the region is too small, reject */
	if *reg_size < 2 {
		return false
	}

	/* re-compute rectangle */
	region2rect(reg, *reg_size, modgrad, reg_angle, prec, p, rec)

	/* re-compute region points density */
	density = float64(*reg_size) / (dist(rec.x1, rec.y1, rec.x2, rec.y2) * rec.width)

	/*------ Second try: reduce region radius ------*/
	if density < density_th {
		return reduce_region_radius(reg, reg_size, modgrad, reg_angle, prec, p,
			rec, used, angles, density_th)
	}

	/* if this point is reached, the density criterion is satisfied */
	return true
}

/*----------------------------------------------------------------------------*/
/*-------------------------- Line Segment Grower   ---------------------------*/
/*----------------------------------------------------------------------------*/

/*----------------------------------------------------------------------------*/

func min(a, b float64) float64 {
	if a < b {
		return a
	}

	return b
}

func points_distance(x1, y1, x2, y2 float64) float64 {
	return math.Sqrt((x2-x1)*(x2-x1) + (y2-y1)*(y2-y1))
}

func rect_distance(a, b *rect, infinity float64, u_a_c, u_b_c *int) float64 {
	a_c := 1
	b_c := 1
	dist11 := points_distance(a.x1, a.y1, b.x1, b.y1)
	dist := dist11
	dist12 := points_distance(a.x1, a.y1, b.x2, b.y2)
	if dist12 < dist {
		dist = dist12
		a_c = 1
		b_c = 2
	}
	dist21 := points_distance(a.x2, a.y2, b.x1, b.y1)
	if dist21 < dist {
		dist = dist21
		a_c = 2
		b_c = 1
	}
	dist22 := points_distance(a.x2, a.y2, b.x2, b.y2)
	if dist22 < dist {
		dist = dist22
		a_c = 2
		b_c = 2
	}

	// check if rectangles are parallel
	if a_c == 1 && b_c == 1 {
		if dist21 > dist22 {
			return infinity
		}
	}
	if a_c == 1 && b_c == 2 {
		if dist22 > dist21 {
			return infinity
		}
	}
	if a_c == 2 && b_c == 1 {
		if dist11 > dist12 {
			return infinity
		}
	}
	if a_c == 2 && b_c == 2 {
		if dist12 > dist11 {
			return infinity
		}
	}
	if u_a_c != nil && u_b_c != nil {
		*u_a_c = a_c
		*u_b_c = b_c
	}
	return dist
}

/*----------------------------------------------------------------------------*/
func rects2rect(reg []int, reg_size int, points []*rect, reg_angle, prec, p float64, rec *rect) {
	if reg_size == 1 {
		*rec = *points[reg[0]]
	}

	/* center of the region:

	   It is computed as the weighted sum of the coordinates
	   of all the centers of the rectangles. The length of the rectangle
	   is used as the weight of a pixel. The sum is as follows
	*/
	x := 0.0
	y := 0.0
	sum := 0.0

	for i := 0; i < reg_size; i++ {
		weight := points[reg[i]].length
		x += float64(points[reg[i]].x) * weight
		y += float64(points[reg[i]].y) * weight
		sum += weight
	}
	x /= sum
	y /= sum

	/* theta */
	theta := get_reg_union_theta(reg, reg_size, x, y, points, reg_angle, prec)
	log.Printf("theta: %v (%v)\n", theta, reg_angle)
	// FIXME!!!!
	//theta = reg_angle

	/* length and width:

	   'l' and 'w' are computed as the distance from the center of the
	   region to pixel i, projected along the rectangle axis (dx,dy) and
	   to the orthogonal axis (-dy,dx), respectively.

	   The length of the rectangle goes from l_min to l_max, where l_min
	   and l_max are the minimum and maximum values of l in the region.
	   Analogously, the width is selected from w_min to w_max, where
	   w_min and w_max are the minimum and maximum of w for the pixels
	   in the region.
	*/
	dx := math.Cos(theta)
	dy := math.Sin(theta)
	l_min := 0.0
	l_max := 0.0
	w_min := 0.0
	w_max := 0.0
	for i := 0; i < reg_size; i++ {
		l := (float64(points[reg[i]].x1)-x)*dx + (float64(points[reg[i]].y1)-y)*dy
		w := -(float64(points[reg[i]].x1)-x)*dy + (float64(points[reg[i]].y1)-y)*dx

		if l > l_max {
			l_max = l
		}
		if l < l_min {
			l_min = l
		}
		if w > w_max {
			w_max = w
		}
		if w < w_min {
			w_min = w
		}

		l = (float64(points[reg[i]].x2)-x)*dx + (float64(points[reg[i]].y2)-y)*dy
		w = -(float64(points[reg[i]].x2)-x)*dy + (float64(points[reg[i]].y2)-y)*dx

		if l > l_max {
			l_max = l
		}
		if l < l_min {
			l_min = l
		}
		if w > w_max {
			w_max = w
		}
		if w < w_min {
			w_min = w
		}
	}

	/* store values */
	rec.x1 = x + l_min*dx
	rec.y1 = y + l_min*dy
	rec.x2 = x + l_max*dx
	rec.y2 = y + l_max*dy
	rec.width = w_max - w_min
	rec.x = x
	rec.y = y
	rec.theta = theta
	rec.dx = dx
	rec.dy = dy
	rec.prec = prec
	rec.p = p

	/* we impose a minimal width of one pixel

	   A sharp horizontal or vertical step would produce a perfectly
	   horizontal or vertical region. The width computed would be
	   zero. But that corresponds to a one pixels width transition in
	   the image.
	*/
	if rec.width < 1.0 {
		rec.width = 1.0
	}
}

func g_region_grow(init_ind int, reg []int, reg_size *int,
	reg_angle *float64, used []bool,
	points []*rect, n int, prec, dist_threshold, length_threshold float64) {
	var a, b int

	/* first points of the region */
	*reg_size = 1
	reg[0] = init_ind
	*reg_angle = math.Atan2(points[init_ind].y2-points[init_ind].y1, points[init_ind].x2-points[init_ind].x1)
	sumdx := 0.0
	sumdy := 0.0
	used[init_ind] = true

	/* try neighbors */
	for i := 0; i < *reg_size; i++ {
		for j := 0; j < n; j++ {
			if used[j] != true &&
				rect_distance(points[reg[i]], points[j], dist_threshold+1, &a, &b) <= dist_threshold {
				if isaligned_(points[j].theta, *reg_angle, prec) {
					/* add neighbor */
					used[j] = true
					reg[*reg_size] = j
					(*reg_size)++

					/* update region's angle */
					dx := points[j].x - points[reg[i]].x
					dy := points[j].y - points[reg[i]].y
					if a == 1 {
						dx *= -1.0
						dy *= -1.0
					}
					sumdx += dx
					sumdy += dy
					*reg_angle = math.Atan2(sumdy, sumdx)
				}
			}
		}
	}
}

type arect []*rect

func (r arect) Len() int {
	return len(r)
}

func (r arect) Less(i, j int) bool {
	return r[i].length < r[i].length
}

func (r arect) Swap(i, j int) {
	r[i], r[j] = r[j], r[i]
}

func prepare_segments(points []*rect, size *int, in *image_double, length_threshold float64) {
	sort.Sort(arect(points))

	i := *size - 1
	for i >= 0 && points[i].length < length_threshold {
		i--
	}

	*size = i + 1
}

func line_segment_grower(img []float64, X, Y int,
	scale, sigma_scale, ang_th float64,
	use_NFA bool, log_eps float64,
	reg_img *[]int, reg_x, reg_y *int,
	out *ntuple_list, xsize, ysize uint,
	region *image_int,
	points []*rect, size int,
	image, angles *image_double,
	length_threshold, dist_threshold float64) {
	ls_count := 0 /* line segments are numbered 1,2,3,... */

	out = new_ntuple_list(7)

	/* angle tolerance */
	prec := math.Pi * ang_th / 180.0
	p := ang_th / 180.0

	prepare_segments(points, &size, image, length_threshold)

	/* Number of Tests - NT

	   The theoretical number of tests is Np.(XY)^(5/2)
	   where X and Y are number of columns and rows of the image.
	   Np corresponds to the number of angle precisions considered.
	   As the procedure 'rect_improve' tests 5 times to halve the
	   angle precision, and 5 more times after improving other factors,
	   11 different precision values are potentially tested. Thus,
	   the number of tests is
	     11 * (X*Y)^(5/2)
	   whose logarithm value is
	     log10(11) + 5/2 * (log10(X) + log10(Y)).
	*/
	logNT := 5.0*(math.Log10(float64(xsize))+math.Log10(float64(ysize)))/2.0 + math.Log10(11.0)

	/* initialize some structures */
	used := make([]bool, size)
	reg := make([]int, size)
	if reg_img != nil && reg_x != nil && reg_y != nil {
		region = new_image_int_ini(image.xsize, image.ysize, 0)
	}

	var reg_size int
	var reg_angle float64
	var rec rect
	var log_nfa float64

	/* search for line segments */
	for j := 0; j < size; j++ {
		if used[j] == false {
			/* find the ~ connected segments */
			g_region_grow(j, reg, &reg_size,
				&reg_angle, used, points, size,
				prec, dist_threshold, length_threshold)

			/* construct rectangular approximation for the region */
			rects2rect(reg, reg_size, points, reg_angle, prec, p, &rec)

			if use_NFA {
				///* compute NFA value */
				log_nfa := rect_improve(&rec, angles, logNT, log_eps)
				log.Println("nfa: %lf %lf\n", log_nfa, log_eps)
				if log_nfa <= log_eps {
					continue
				}
			}

			/* A New Line Segment was found! */
			ls_count++ /* increase line segment counter */

			/* scale the result values if a subsampling was performed */
			if scale != 1.0 {
				rec.x1 /= scale
				rec.y1 /= scale
				rec.x2 /= scale
				rec.y2 /= scale
				rec.width /= scale
			}

			/* add line segment found to output */
			add_7tuple(out, rec.x1, rec.y1, rec.x2, rec.y2,
				rec.width, rec.p, log_nfa)

			///* add region number to 'region' image if needed */
			//if( region != nil )
			//for(i=0; i<reg_size; i++)
			//region.data[ reg[i].x + reg[i].y * region.xsize ] = ls_count
		}
	}
}

/*----------------------------------------------------------------------------*/
/*-------------------------- Line Segment Detector ---------------------------*/
/*----------------------------------------------------------------------------*/

/*----------------------------------------------------------------------------*/
/** LSD full interface.
 */
func LineSegmentDetection(img []float64, X, Y int,
	scale, sigma_scale, quant, ang_th, log_eps, density_th, union_ang_th float64,
	union_use_NFA bool, union_log_eps float64,
	n_bins int, need_to_union bool,
	reg_img *[]int, reg_x, reg_y *int,
	length_threshold, dist_threshold float64) ([]float64, int) {
	var list_p *coorlist
	var region *image_int
	out := new_ntuple_list(7)
	ls_count := 0 /* line segments are numbered 1,2,3,... */

	/* check parameters */
	if img == nil || X <= 0 || Y <= 0 {
		fatal_error("invalid image input.")
	}
	if scale <= 0.0 {
		fatal_error("'scale' value must be positive.")
	}
	if sigma_scale <= 0.0 {
		fatal_error("'sigma_scale' value must be positive.")
	}
	if quant < 0.0 {
		fatal_error("'quant' value must be positive.")
	}
	if ang_th <= 0.0 || ang_th >= 180.0 {
		fatal_error("'ang_th' value must be in the range (0,180).")
	}
	if density_th < 0.0 || density_th > 1.0 {
		fatal_error("'density_th' value must be in the range [0,1].")
	}
	if n_bins <= 0 {
		fatal_error("'n_bins' value must be positive.")
	}

	var angles, modgrad *image_double
	var rec rect
	var rects []*rect

	/* angle tolerance */
	prec := math.Pi * ang_th / 180.0
	p := ang_th / 180.0
	rho := quant / math.Sin(prec) /* gradient magnitude threshold */

	/* load and scale image (if necessary) and compute angle at each pixel */
	image := new_image_double_ptr(uint(X), uint(Y), img)
	if scale != 1.0 {
		scaled_image := gaussian_sampler(image, scale, sigma_scale)
		angles = ll_angle(scaled_image, rho, &list_p,
			&modgrad, uint(n_bins))
	} else {
		angles = ll_angle(image, rho, &list_p, &modgrad, uint(n_bins))
	}
	xsize := angles.xsize
	ysize := angles.ysize

	/* Number of Tests - NT

	   The theoretical number of tests is Np.(XY)^(5/2)
	   where X and Y are number of columns and rows of the image.
	   Np corresponds to the number of angle precisions considered.
	   As the procedure 'rect_improve' tests 5 times to halve the
	   angle precision, and 5 more times after improving other factors,
	   11 different precision values are potentially tested. Thus,
	   the number of tests is
	     11 * (X*Y)^(5/2)
	   whose logarithm value is
	     log10(11) + 5/2 * (log10(X) + log10(Y)).
	*/
	logNT := 5.0*(math.Log10(float64(xsize))+math.Log10(float64(ysize)))/2.0 + math.Log10(11.0)
	min_reg_size := int(-logNT / math.Log10(p)) /* minimal number of points in region
	   that can give a meaningful event */

	/* initialize some structures */
	if reg_img != nil && reg_x != nil && reg_y != nil { /* save region data */
		region = new_image_int_ini(angles.xsize, angles.ysize, 0)
	}
	used := new_image_bool_ini(xsize, ysize, false)
	reg := make([]point, xsize*ysize)

	var reg_size int
	var reg_angle float64

	/* search for line segments */
	for ; list_p != nil; list_p = list_p.next {
		if used.data[list_p.x+list_p.y*int(used.xsize)] == false &&
			angles.data[list_p.x+list_p.y*int(angles.xsize)] != NOTDEF {
			/* there is no risk of double comparison problems here
			   because we are only interested in the exact NOTDEF value */

			/* find the region of connected point and ~equal angle */
			region_grow(list_p.x, list_p.y, angles, reg, &reg_size,
				&reg_angle, used, prec)

			/* reject small regions */
			if reg_size < min_reg_size {
				continue
			}

			/* construct rectangular approximation for the region */
			region2rect(reg, reg_size, modgrad, reg_angle, prec, p, &rec)

			/* Check if the rectangle exceeds the minimal density of
			   region points. If not, try to improve the region.
			   The rectangle will be rejected if the final one does
			   not fulfill the minimal density condition.
			   This is an addition to the original LSD algorithm published in
			   "LSD: A Fast Line Segment Detector with a False Detection Control"
			   by R. Grompone von Gioi, J. Jakubowicz, J.M. Morel, and G. Randall.
			   The original algorithm is obtained with density_th = 0.0.
			*/
			if !refine(reg, &reg_size, modgrad, reg_angle, prec, p, &rec, used, angles, density_th) {
				continue
			}

			/* compute NFA value */
			log_nfa := rect_improve(&rec, angles, logNT, log_eps)
			if log_nfa <= log_eps {
				continue
			}

			/* A New Line Segment was found! */
			ls_count++ /* increase line segment counter */

			/*
			   The gradient was computed with a 2x2 mask, its value corresponds to
			   points with an offset of (0.5,0.5), that should be added to output.
			   The coordinates origin is at the center of pixel (0,0).
			*/
			rec.x1 += 0.5
			rec.y1 += 0.5
			rec.x2 += 0.5
			rec.y2 += 0.5

			rec.length = math.Sqrt((rec.x1-rec.x2)*(rec.x1-rec.x2) + (rec.y1-rec.y2)*(rec.y1-rec.y2))
			rec.x1 = float64(int(rec.x1))
			rec.y1 = float64(int(rec.y1))
			rec.x2 = float64(int(rec.x2))
			rec.y2 = float64(int(rec.y2))
			//printf("length: %lf\n", rec.length)
			rects = append(rects, &rec)

			/* scale the result values if a subsampling was performed */
			if scale != 1.0 {
				rec.x1 /= scale
				rec.y1 /= scale
				rec.x2 /= scale
				rec.y2 /= scale
				rec.width /= scale
				//rec.length /= scale
			}

			/* add line segment found to output */
			add_7tuple(out, rec.x1, rec.y1, rec.x2, rec.y2,
				rec.width, rec.p, log_nfa)

			/* add region number to 'region' image if needed */
			if region != nil {
				for i := 0; i < reg_size; i++ {
					region.data[reg[i].x+reg[i].y*int(region.xsize)] = ls_count
				}
			}
		}
	}

	if need_to_union {
		line_segment_grower(img, X, Y, scale, sigma_scale,
			union_ang_th, union_use_NFA, union_log_eps,
			reg_img, reg_x, reg_y, out,
			xsize, ysize, region, rects, len(rects),
			image, angles, length_threshold, dist_threshold)
	}

	/* return the result */
	if reg_img != nil && reg_x != nil && reg_y != nil {
		if region == nil {
			fatal_error("'region' should be a valid image.")
		}
		*reg_img = region.data
		if region.xsize > uint(math.MaxInt64) || region.ysize > uint(math.MaxInt64) {
			fatal_error("region image to big to fit in INT sizes.")
		}
		*reg_x = int(region.xsize)
		*reg_y = int(region.ysize)
	}

	if out.size > uint(math.MaxInt64) {
		fatal_error("too many detections to fit in an INT.")
	}

	return out.values, int(out.size)
}

/*----------------------------------------------------------------------------*/
/** LSD Simple Interface with Scale and Region output.
 */
func LSD_scale_region(img []float64, X, Y int, scale float64,
	reg_img *[]int, reg_x, reg_y *int) ([]float64, int) {
	/* LSD parameters */
	sigma_scale := 0.6 /* Sigma for Gaussian filter is computed as
	   sigma = sigma_scale/scale.                    */
	quant := 2.0 /* Bound to the quantization error on the
	   gradient norm.                                */
	ang_th := 22.5    /* Gradient angle tolerance in degrees.           */
	log_eps := 0.0    /* Detection threshold: -log10(NFA) > log_eps     */
	density_th := 0.7 /* Minimal density of region points in rectangle. */
	n_bins := 1024    /* Number of bins in pseudo-ordering of gradient
	   modulus.                                       */

	// FIXME: get good initial parametrs
	need_to_union := false /* Bool if you need to post process image by union close segments */
	union_ang_th := 7.0
	union_use_NFA := false
	union_log_eps := 0.0
	length_threshold := 5.0 /* Minimum length of segment to union     */
	dist_threshold := 5.0   /* Maximum distance between two line which we would union */

	return LineSegmentDetection(img, X, Y, scale, sigma_scale, quant,
		ang_th, log_eps, density_th, union_ang_th,
		union_use_NFA, union_log_eps, n_bins,
		need_to_union, reg_img, reg_x, reg_y,
		length_threshold, dist_threshold)
}

/*----------------------------------------------------------------------------*/
/** LSD Simple Interface with Scale.
 */
func LSD_scale(img []float64, X, Y int, scale float64) ([]float64, int) {
	return LSD_scale_region(img, X, Y, scale, nil, nil, nil)
}

/*----------------------------------------------------------------------------*/
/** LSD Simple Interface.
 */
func LSD(img []float64, X, Y int) ([]float64, int) {
	// LSD parameters
	scale := 0.8 // Scale the image by Gaussian filter to 'scale'.

	return LSD_scale(img, X, Y, scale)
}

/*----------------------------------------------------------------------------*/
