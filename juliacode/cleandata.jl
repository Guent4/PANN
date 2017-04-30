using DataFrames #to read in data
using PyPlot #to plot data
using CurveFit #to fit curves

#df =  readtable("../dating/data.csv")
#showcols(df) #lets us see a summary showing missing data

function fillInNan(df)
	num_people, num_cols = size(df)
	threshold_percent = .55
	threshold = threshold_percent*num_people
	j = 1
	toDelete =Int64[]
	while(j<num_cols)
		num_NA = size(df[isna(df[j]),j],1)
		#@show num_NA
		if(num_NA>threshold)
			push!(toDelete, j);
			#num_cols = num_cols -1	
		end
		j = j+1
	end
	delete!(df, toDelete);
	j = 1
	types=eltypes(df)
	toDelete =Int64[]
	num_people, num_cols = size(df)
	while(j<num_cols)
		#@show j, types[j]
		if(types[j]==String)
			#delete!(df,j);
			push!(toDelete, j);
		end
		j = j+1
	end
	delete!(df, toDelete);
	@show size(df)
	num_people, num_cols = size(df)
	types=eltypes(df)
	for j in collect(1:num_cols)
		@show j, types[j]
		curr_col = df[!isna(df[j]),j] #get data with NAs removed
		if(types[j]==Float64)
			if(typeof(curr_col[1])==Float64)
				fill_val = mean(curr_col) #calculate the mean
			elseif(typeof(curr_col[1])==Int64)
				fill_val=Int(round(mean(curr_col)))
			else
				temp = Float64[]
				for k in size(curr_col,1)
					@show replace(curr_col[k], ",", "")
					t= parse(Float64, replace(curr_col[k], ",", ""))
					@show t, typeof(t)
					push!(temp, t)
				end
				@show temp
				fill_val = string(mean(temp))
			end
			df[isna(df[j]),j]=fill_val #fill NAs with mean value
		elseif(types[j]==Int64)
			fill_val = mean(curr_col) #calculate the mean
			df[isna(df[j]),j]=Int(round(fill_val)) #fill NAs with rounded mean value
		end
	end
	#remove id number data
	delete!(df, :id);
	delete!(df, :iid);
	delete!(df, :idg);
	delete!(df, :position);
	delete!(df, :positin1);
	delete!(df, :order);
	delete!(df, :partner);
	delete!(df, :pid);
	@show showcols(df)
	writetable("../dating/cleanedData.csv",df)
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

function compareAttractiveness(df)
	close("all")
	ids = unique(df[:iid])
	selfrates = Float64[]
#	otherrate = df[:attr]
	other_rates =Float64[]

	for id in ids
		other_rate = mean(df[df[:iid].==id, :attr])
		@show other_rate
		push!(other_rates, other_rate)
		my_rate = (df[df[:iid].==id, :attr3_1])[1]
		@show my_rate
		push!(selfrates, my_rate)
	end

	plot(selfrates, other_rates, "k.")
	axis([1,11,1,11])
	xlabel("Personal Attractiveness Rating", fontsize = 18)
	ylabel("Average Attractiveness Rating by Others", fontsize = 18)

	figure(figsize=[15,15])
	self_counts = countmap(selfrates)
	other_counts =countmap(other_rates)
	for key in keys(self_counts)
		bar(key, self_counts[key], color=".5", alpha = .5)
	end

	x,y=hist(other_rates, 20)
	@show x
	@show y
	bar(x[2:end],y, color = "c", alpha = .5, align="center", width = .4)
	xlabel("Attractiveness", fontsize=18)
	ylabel("Number of Participants", fontsize=18)
	title("Comparsion of Self Assesed Attractivess to Attractiveness Assesed by Others", fontsize=18)
	savefig("../figures/attractivenessbar.pdf")


end

function correlateAttractivenessAndMatches(df)
	close("all")
	figure(figsize=[15,15])
	ids = unique(df[:iid])
	numMatches = Float64[]
	selfrates = Float64[]
	other_rates =Float64[]
	for id in ids
		match_count = sum(df[df[:iid].==id, :match])
		@show match_count
		push!(numMatches, match_count)
		other_rate = mean(df[df[:iid].==id, :attr])
		@show other_rate
		push!(other_rates, other_rate)
		my_rate = (df[df[:iid].==id, :attr3_1])[1]
		@show my_rate
		push!(selfrates, my_rate)
	end
	plot(other_rates, numMatches, "kx")
	a,b= linear_fit(other_rates, numMatches)
	x_plot = collect(1:.1:10)
	y_plot = a*x_plot+b
	@show a, b, y_plot
	plot(x_plot, y_plot, "c-")
	xlabel("Averaged Attractiveness, as assessed by others", fontsize = 18)
	ylabel("Number of Matches", fontsize = 18)
	annotate(string("y=", a,"x+", b), 	
		xy=[.65;.8],
		xycoords="axes fraction",
		xytext=[-10,10],
		textcoords="offset points",
		fontsize=18.0,
		ha="right",
		va="bottom")
	savefig("../figures/linestNumMatchesAndAttractiveness.pdf")
end

function plotMatching(df)
	all_ids = df[:iid]
	num_rows, num_cols = size(df)
	results = zeros(maximum(all_ids), maximum(all_ids))
	waves = collect(1:21)
	lower = 1
	upper = 1
	processed = 0
	#white = no data, 
	#dark red match, light red one person interested, orange, no one interested
	for currwave in waves
		close("all")
		figure(figsize=[20,20])
		waveids = (df[df[:wave].==currwave,:iid])
		num_rows_in_wave = size(waveids,1)
		waveids = unique(waveids)
		num_in_wave = size(waveids,1)
		results = fill(-1.0,num_in_wave+1, num_in_wave+1)
		upper = lower+num_rows_in_wave
		@show lower, upper
		for k in collect(lower:upper)
			for j in collect(lower:upper)
				my_id_1= df[:iid][j]
				partner_id_1 = df[:pid][j]
				my_id_2= df[:iid][k]
				partner_id_2 = df[:pid][k]
				if(my_id_1==partner_id_2) #if person one is talking to person 2
					if((df[:dec][j]==df[:dec][k])&& df[:dec][j]==1) #we have a match
						@show my_id_1-processed, my_id_2-processed, num_in_wave, processed
						results[my_id_1-processed, my_id_2-processed]=1
						@printf("found match between %i and %i in wave %i\n", my_id_1, my_id_2, currwave)
					elseif(df[:dec][j]==1 && df[:dec][k]==0 ||(df[:dec][k]==1 && df[:dec][j]==0))
						results[my_id_1-processed, my_id_2-processed] = .5 #unrequited match
					elseif((df[:dec][j]==df[:dec][k])&& df[:dec][j]==0) #mutal unattraction 
						results[my_id_1-processed, my_id_2-processed]=0.0
					end
				end

			end
		end
	lower = lower+num_rows_in_wave
	processed = processed + num_in_wave
	pcolormesh(results, cmap = "Reds")
	#axis([0,num_in_wave+1, 0, num_in_wave+1])
	savefig(string("../figures/MatchesInWave", currwave, ".pdf"))
	end
	return results		
end
