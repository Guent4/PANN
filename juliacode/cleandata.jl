using DataFrames #to read in data

df =  readtable("../dating/data.csv")
showcols(df) #lets us see a summary showing missing data

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
