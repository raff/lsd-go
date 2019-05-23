package main

import (
	"flag"
	"fmt"
	"image"
	_ "image/gif"
	_ "image/jpeg"
	_ "image/png"
	"math"
	"os"

	"github.com/fogleman/gg"
	"github.com/raff/lsd-go"
)

func makeGray(ima image.Image) ([]float64, int, int) {
	bounds := ima.Bounds()
	width := bounds.Dx()
	height := bounds.Dy()

	gray := make([]float64, width*height)
	gp := 0

	for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
		for x := bounds.Min.X; x < bounds.Max.X; x++ {
			//fmt.Println("x", x, "y", y, "gp", gp, "size", len(gray))
			c := ima.At(x, y)
			r, g, b, _ := c.RGBA()
			gray[gp] = 0.299*float64(r) + 0.587*float64(g) + 0.114*float64(b)
			gp++
		}
	}

	return gray, width, height
}

func main() {
	scale := flag.Float64("scale", 0.8, "Scale image by Gaussian filter before processing.")
	sigma := flag.Float64("sigma", 0.6, "Sigma for Gaussian filter is computed as sigma_coef/scale.")
	quant := flag.Float64("quant", 2.0, "Bound to quantization error on the gradient norm.")
	theta := flag.Float64("theta", 22.5, "Gradient angle tolerance in degrees.")
	eps := flag.Float64("eps", 0.0, "Detection threshold, -log10(max. NFA)")
	density := flag.Float64("density", 0.7, "Minimal density of region points in a rectangle to be accepted.")
	bins := flag.Int("bins", 1024, "Number of bins in 'ordering' of gradient modulus.")
	union := flag.Bool("union", false, "Do you need to post process image by merging close segments")
	utheta := flag.Float64("union-theta", 0.0, "Union angle tolerance in degrees.")
	ueps := flag.Float64("union-eps", -1.0, "Union detection threshold")
	ulength := flag.Float64("union-length", 5.0, "Minimum length of segment to union.")
	udist := flag.Float64("union-distance", 5.0, "Maximum distance between two line which we would union.")
	outf := flag.String("format", "verbose", "Output format (verbose, csv, image)")
	color := flag.String("color", "#C0C0C0", "RGB color to use to highlight lines")
	length := flag.Float64("length", 0, "Minimum line length")

	flag.Parse()

	if flag.NArg() == 0 {
		fmt.Println("usage: lds [options] image-file...")
		return
	}

	for _, arg := range flag.Args() {
		f, err := os.Open(arg)
		if err != nil {
			fmt.Println(err)
			break
		}

		ima, format, err := image.Decode(f)
		f.Close()

		if err != nil {
			fmt.Println("Decode", arg, err)
			break
		}

		fmt.Printf("%v: %v %vx%v\n", arg, format, ima.Bounds().Dx(), ima.Bounds().Dy())

		gray, width, height := makeGray(ima)

		segs, n := lsd.LineSegmentDetection(gray, width, height,
			*scale,
			*sigma,
			*quant,
			*theta,
			*eps,
			*density,
			*utheta,
			*ueps > 0.0,
			*ueps,
			*bins,
			*union,
			nil,
			nil,
			nil,
			*ulength,
			*udist)

		l := 0

		switch *outf {
		case "image":
			gc := gg.NewContextForImage(ima)
			gc.SetLineCap(gg.LineCapSquare)
			gc.SetHexColor(*color)

			/*
			   gc.DrawRectangle(
			       float64(ima.Bounds().Min.X+1),
			       float64(ima.Bounds().Min.Y+1),
			       float64(ima.Bounds().Max.X-1),
			       float64(ima.Bounds().Max.Y-1))
			   gc.SetLineWidth(4)
			   gc.Stroke()
			*/

			// x1,y1,x2,y2,width,p,nfa
			for i := 0; i < n; i++ {
				x1, y1, x2, y2, width := segs[l+0], segs[l+1], segs[l+2], segs[l+3], segs[l+4]

				// fmt.Println(i, "x1", x1, "x2", x2, "y1", y1, "y2", y2)

				w, h := math.Abs(x2-x1), math.Abs(y2-y1)

				if w != 0.0 && w < *length {
					goto continue_line
				}

				if h != 0.0 && h < *length {
					goto continue_line
				}

				gc.DrawLine(x1, y1, x2, y2)
				gc.SetLineWidth(width)
				gc.Stroke()
			continue_line:
				l += 7
			}

			gc.SavePNG("lines.png")

		case "csv":
			fmt.Println("x1,y1,x2,y2,width,p,nfa")
			for i := 0; i < n; i++ {
				fmt.Printf("%f,%f,%f,%f,%f,%f,%f\n",
					segs[l+0], segs[l+1], segs[l+2], segs[l+3], segs[l+4], segs[l+5], segs[l+6])

				l += 7
			}

		default:
			for i := 0; i < n; i++ {
				x1, y1, x2, y2 := segs[l+0], segs[l+1], segs[l+2], segs[l+3]

				fmt.Printf("%4.0d %2d %2d: x1:%6.1f y1:%6.1f x2:%6.1f y2:%6.1f w:%3.1f p:%3.2f nfa:% 5.1f\n",
					i+1, int(math.Abs(x2-x1)), int(math.Abs(y2-y1)),
					segs[l+0], segs[l+1], segs[l+2], segs[l+3], segs[l+4], segs[l+5], segs[l+6])

				l += 7
			}
		}
	}
}
