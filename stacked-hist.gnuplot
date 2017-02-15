#!/usr/bin/gnuplot

set term postscript eps enhanced color
#set output 'stacked-hist.eps'
set output ARG1.'.eps'

#set title "LUBM TransE Tail Predictions Unfiltered"
set title ARG1
set key top left outside horizontal autotitle columnhead

#set xtics rotate by 90 offset 0,-5 out nomirror
set autoscale x
set ytics out nomirror

set style fill solid border -1
set boxwidth 0.5 relative
set style data histograms
set style histogram rowstacked
set xlabel "Degree of node"

plot ARG1 using 2, '' using 3:xtic(1)
