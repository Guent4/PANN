using DataFrames #to read in data
using PyPlot #to plot data

#df =  readtable("../dating/data.csv")
#showcols(df) #lets us see a summary showing missing data

function fillInNan(df)
	num_people, num_cols = size(df)
	types=eltypes(df)
	for j in collect(1:num_cols)
		curr_col = df[!isna(df[j]),j] #get data with NAs removed
		if(types[j]==Float64)
			fill_val = mean(curr_col) #calculate the mean
			df[isna(df[j]),j]=fill_val #fill NAs with mean value
		elseif(types[j]==Int64)
			fill_val = mean(curr_col) #calculate the mean
			df[isna(df[j]),j]=Int(round(fill_val)) #fill NAs with rounded mean value
		end
	end
	return df
end

function makeFreqGraphs(df)
	close("all")
	agefreq = countmap(df[!isna(df[:age]), :age])
	for key in keys(agefreq)
		bar(key, agefreq[key], color=".5")
	end
	xlabel("Age", fontsize = 18)
	ylabel("Number of Participants", fontsize = 18)
	title("Distribution of Ages", fontsize = 24)
	savefig("figures/AgeFreq.pdf")
	expnumfreq = countmap(df[!isna(df[:expnum]), :expnum])
	figure()
	for key in keys(expnumfreq)
		bar(key, expnumfreq[key], color=".5")
	end
	xlabel("How many people will be interested in dating you", fontsize = 18)
	ylabel("Number of Participants", fontsize = 18)
	title("Number of People Interested in You", fontsize = 24)
	axis([0,20,0,300])
	savefig("figures/ExpNumFreq.pdf")
end

function makeWhatMattersToMeGraph(df)
	close("all")
	figure(figsize=[20,10])
	dependsonwave =[:attr1_1, :sinc1_1, :intel1_1, :fun1_1, :amb1_1, :shar1_1]
	names = ["Attractiveness", "Sincerity", "Intelligence", "Fun", "Ambition", "Has shared interests/hobbies"]
	cdf = df
	if cdf[:wave] in collect(6:9) #if we're in the waves rate 1-10, multiply by 10 to scale
		for item in dependsonwave
			cdf[item] = cdf[item]*10
		end
	end

	for item in dependsonwave
		#figure()
		@show item
#		freq = countmap(cdf[!isna(df[item]), item])
#		for key in keys(freq)
#			bar(key, freq[key], alpha = .5)
#		end
		plot(cdf[:iid], cdf[item], ".")
	end
	axis([128,220,0,45])
	legend(names)
	xlabel("Subject ID", fontsize=18)
	ylabel("Importance", fontsize=18)
	savefig("figures/WhatYouLookForInOpSex.pdf")
	
end
