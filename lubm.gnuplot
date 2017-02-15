#!/usr/bin/gnuplot
reset
#set terminal latex
set term postscript eps enhanced
set output "lubm.eps"

power(x,y) = x ** y

#set logscale x 2
#set xrange [power(2,18):power(2,24)]
#set xrange [1:8000]
set xrange [1:8]
set xlabel "Degree"

set ylabel "Hit/Miss"
set yrange [0:2.5]

set title "LUBM Hit-Map"
set key reverse Left outside
set grid

# Style could be linespoints
set style data points

# brackets are necessary for "using" directive:
#   (some_fun()):2:3:... where 2 and 3 are columns from data file

plot 'lubm1_uri-tail-predictions-unfiltered.txt' using 1:2 title 'Tail predictions Unfiltered' lc rgb 'red', \
'lubm1_uri-tail-predictions-filtered.txt' using 1:2 title 'Tail predictions Filtered' lc rgb 'orange'
#'lubm1_uri-head-predictions-unfiltered.txt' using 1:2 title 'Head predictions Unfiltered' lc rgb 'green', \
#'lubm1_uri-head-predictions-filtered.txt' using 1:2 title 'Head predictions Filtered' lc rgb 'blue'
