using PyPlot
using ExcelReaders
using DataFrames

function makePlot(pathToData, cells, xlabelstr)
	close("all")
	@show cells
	data = (readxl(pathToData, cells))
	xdata = data[:,1]
	timeSeq = data[:,2]
	timePar1=data[:,3]
	timePar4=data[:,4]
	@show xdata
	@show timeSeq
	@show timePar1

	figure(figsize=[15,15])
	plot(xdata, timeSeq./timePar1, "kd-", linewidth = 2.0)
	plot(xdata, timeSeq./timePar4, color =".5", marker = "o", linewidth = 2.0)
	legend(["1 process", "4 processes"], loc ="best")
	xlabel(xlabelstr, fontsize = 20)
	ylabel("Speed Up Over Serial", fontsize = 20)
	titlestr = string("Effect of Changing ", xlabelstr)
	title(titlestr, fontsize = 24)
	ax = gca()
	ax[:tick_params](labelsize=20)
	savefig(string("../figures/", titlestr, ".pdf"))	
end

function makeAllPlots()
	pathToData = "../performancedata/ECE5720FinalData.xlsx"
	changingN = "Sheet1!A6:F10"
	changingLayerSize = "Sheet1!A19:F23"
	changingNumLayers = "Sheet1!A32:F36"

	cells = [changingN, changingLayerSize, changingNumLayers]
	xlabels= ["N", "Layer Size", "Number of Layers"]
	
	j = 1
	for c in cells
		makePlot(pathToData, c, xlabels[j])
		j = j+1
	end
end
