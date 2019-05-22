package main

import (
	"fmt"

	"github.com/raff/lsd-go"
)

/* create a simple image: half black, half gray */
func make_image(X, Y int, horiz bool) []float64 {
	image := make([]float64, X*Y)

	for x := 0; x < X; x++ {
		for y := 0; y < Y; y++ {
			if horiz {
				if y < Y/2 {
					image[x+y*X] = 0.0
				} else {
					image[x+y*X] = 64.0
				}
			} else {
				if x < X/2 {
					image[x+y*X] = 0.0
				} else {
					image[x+y*X] = 64.0
				}
			}
		}
	}

	return image
}

func main() {
	X := 128 /* x image size */
	Y := 128 /* y image size */

	/* create a simple image: left half black, right half gray */
	image := make_image(X, Y, false)
	//fmt.Println(image)

	/* LSD call */
	out, n := lsd.LSD(image, X, Y)

	/* print output */
	fmt.Printf("%d line segments found:\n", n)

	for i := 0; i < n; i++ {
		l := i * 7
		fmt.Printf("x1:%4.1f y1:%4.1f x2:%4.1f y2:%4.1f w:%3.1f p:%3.2f nfa:%5.1f\n",
			out[l+0], out[l+1], out[l+2], out[l+3], out[l+4], out[l+5], out[l+6])
	}

	/* create a simple image: top half black, bottom half gray */
	image = make_image(X, Y, true)
	//fmt.Println(image)

	/* LSD call */
	out, n = lsd.LSD(image, X, Y)

	/* print output */
	fmt.Printf("%d line segments found:\n", n)

	for i := 0; i < n; i++ {
		l := i * 7
		fmt.Printf("x1:%4.1f y1:%4.1f x2:%4.1f y2:%4.1f w:%3.1f p:%3.2f nfa:%5.1f\n",
			out[l+0], out[l+1], out[l+2], out[l+3], out[l+4], out[l+5], out[l+6])
	}
}
