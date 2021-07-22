library("WaveletComp")
library("signal")
#install.packages("pracma")
library("pracma")
#install.packages("biwavelet")
library("biwavelet")
#install.packages("seewave")
library("seewave")
library("ggplot2")

sequences_df <- data.frame()

max_cropping_ratio = 0.20
number_of_slices = 0

file_ext = '_ankles'

all_data <- read.csv(file=paste(c('gait_values', file_ext, '.csv'), sep="", collapse=""),  sep=",",dec=".",header=T)
i = 0
for (sequence_id in unique(all_data$sequence_id)) {
  seq_id = unique(all_data$sequence_id)[i+1]
  i = i + 1
  cat(c(round((i/length(unique(all_data$sequence_id)))*100, digits = 1)), "%\n")
  sequence <- all_data
  for (slice_begin in (0:number_of_slices)) {
    for (slice_end in (0:number_of_slices)) {
      # Take only the current sequence
      sequence <- sapply(all_data, function(x) x[which(all_data$sequence_id == seq_id)])
      sequence <- data.frame(sequence)
      
      if (number_of_slices > 0) {
        # Compute the start and end indexes for the cropping
        start_index = (slice_begin/number_of_slices) * (max_cropping_ratio * length(sequence$pir0))
        start_index = as.integer(start_index)
        ending_index = length(sequence$pir0) - (slice_end/number_of_slices) * (max_cropping_ratio * length(sequence$pir0))
        ending_index = as.integer(ending_index)
      } else {
        start_index = 0
        ending_index = length(sequence$pir0)
      }
      
      # Crop the sequence
      sequence <- sequence[start_index:ending_index, ]
      compteur <- 0:(length(sequence$pir0)-1)
      tryCatch({
        
        # Compute the wavelet transform from the 4 sensors
        data_pir0 <- data.frame(index=compteur, pir0=sequence$pir0)
        data_pir1 <- data.frame(index=compteur, pir1=sequence$pir1)
        data_pir2 <- data.frame(index=compteur, pir2=sequence$pir2)
        data_pir3 <- data.frame(index=compteur, pir3=sequence$pir3)
        
        # dt = 1/3.200 # Change the "period" scale
        dt = 1/40
        dj = 1/100 # Change the "resolution" of the computation -> 1 = fucked up, 0.001 = very precise
        n.sim = 2 # Used to compute a p-value, but not useful here (below, we set make.pval to False)
        
        my.w0 = analyze.wavelet(data_pir0, "pir0",
                                method = "white.noise",
                                loess.span = 0,
                                dt = dt, dj = dj, n.sim = n.sim, verbose = FALSE,
                                make.pval = FALSE)
        my.w1 = analyze.wavelet(data_pir1, "pir1",
                                method = "white.noise",
                                loess.span = 0,
                                dt = dt, dj = dj, n.sim = n.sim, verbose = FALSE,
                                make.pval = FALSE)
        my.w2 = analyze.wavelet(data_pir2, "pir2",
                                method = "white.noise",
                                loess.span = 0,
                                dt = dt, dj = dj, n.sim = n.sim, verbose = FALSE,
                                make.pval = FALSE)
        my.w3 = analyze.wavelet(data_pir3, "pir3",
                                method = "white.noise",
                                loess.span = 0,
                                dt = dt, dj = dj, n.sim = n.sim, verbose = FALSE,
                                make.pval = FALSE)
        wt.image(my.w3,
                 color.key = "interval",
                 plot.contour = F, plot.ridge = F,
                 n.levels = 64,
                 legend.params = list(lab = "wavelet power levels", mar = 4.7),
                 periodlab = "period (ms)", timelab = "time (ms)")
        
        # Compute the peak to get the period and the time
        maxPeriod0 = my.w0$Period[which(my.w0$Ampl == max(my.w0$Ampl), arr.ind = TRUE)[1, 'row']]
        maxTime0 = unname(which(my.w0$Ampl == max(my.w0$Ampl), arr.ind = TRUE)[1, 'col']) / length(compteur)
        maxAmpl0 = max(my.w0$Ampl)
        maxPeriod1 = my.w1$Period[which(my.w1$Ampl == max(my.w1$Ampl), arr.ind = TRUE)[1, 'row']]
        maxTime1 = unname(which(my.w1$Ampl == max(my.w1$Ampl), arr.ind = TRUE)[1, 'col']) / length(compteur)
        maxAmpl1 = max(my.w1$Ampl)
        maxPeriod2 = my.w2$Period[which(my.w2$Ampl == max(my.w2$Ampl), arr.ind = TRUE)[1, 'row']]
        maxTime2 = unname(which(my.w2$Ampl == max(my.w2$Ampl), arr.ind = TRUE)[1, 'col']) / length(compteur)
        maxAmpl2 = max(my.w2$Ampl)
        maxPeriod3 = my.w3$Period[which(my.w3$Ampl == max(my.w3$Ampl), arr.ind = TRUE)[1, 'row']]
        maxTime3 = unname(which(my.w3$Ampl == max(my.w3$Ampl), arr.ind = TRUE)[1, 'col']) / length(compteur)
        maxAmpl3 = max(my.w3$Ampl)
        
        # Compute the difference from the two sensors on the same side
        delta_time_0_2 = maxTime2 - maxTime0
        delta_time_1_3 = maxTime3 - maxTime1
        
        speed = mean(sequence$walking_speed)
        direction = sequence$direction[1]
        corridor = sequence$corridor[1]
        corridor_position = sequence$corridor_position[1]
        nb_steps_motion = sequence$nb_steps_motion[1]
        nb_steps_distance = sequence$nb_steps_distance[1]
        cycle_time = sequence$cycle_time[1]
        cadence = sequence$cadence[1]
        stance_ratio_right = sequence$stance_ratio_right[1]
        stance_ratio_left = sequence$stance_ratio_left[1]
        double_support_ratio = sequence$double_support_ratio[1]
        footstep_width = sequence$footstep_width[1]
        stride_length = sequence$stride_length[1]
        stride_length_right = sequence$stride_length_right[1]
        stride_length_left = sequence$stride_length_left[1]
        if (is.null(direction)) {
          direction = -2
        }
        if (is.null(corridor)) {
          corridor = -2
        }
        
        # Add the results in a new row of the DataFrame
        sequences_df <- rbind(sequences_df, data.frame(
          sequence_id = sequence_id,
          period_pir0 = maxPeriod0,
          time_pir0 = maxTime0,
          ampl_pir0 = maxAmpl0,
          period_pir1 = maxPeriod1,
          time_pir1 = maxTime1,
          ampl_pir1 = maxAmpl1,
          period_pir2 = maxPeriod2,
          time_pir2 = maxTime2,
          ampl_pir2 = maxAmpl2,
          period_pir3 = maxPeriod3,
          time_pir3 = maxTime3,
          ampl_pir3 = maxAmpl3,
          delta_time_0_2 = delta_time_0_2,
          delta_time_1_3 = delta_time_1_3,
          walking_speed = speed,
          direction = direction,
          corridor = corridor,
          corridor_position = corridor_position,
          nb_steps_motion = nb_steps_motion,
          nb_steps_distance = nb_steps_distance,
          cycle_time = cycle_time,
          cadence = cadence,
          stance_ratio_right = stance_ratio_right,
          stance_ratio_left = stance_ratio_left,
          double_support_ratio = double_support_ratio,
          footstep_width = footstep_width,
          stride_length = stride_length,
          stride_length_right = stride_length_right,
          stride_length_left = stride_length_left
        ))
      }, error= function(err){
        cat(c("Erreur survenue sur la sequence #", seq_id, "(sequence de taille ", length(compteur), ")\n"))
      })
    }
  }
} 

# Save the dataframe as a CSV
write.csv(sequences_df, paste(c("sequences_resumed", file_ext, ".csv"), sep = "", collapse = ""))
  # 
  # 
  # get_peak <- function(indata) {
  #   data <- as.numeric(indata)
  #   peak_val <- findpeaks(data)
  #   #peak <- peak_val[,1]
  #   return (peak_val)
  # }
  # 
  # get_width <- function(peak){
  #   t1 <- peak[,3]
  #   t2 <- peak[,4]
  #   width <- t2 - t1
  #   return (width)
  # }
  # 
  # 
  # p <- get_peak(gest3$pir0)
  # print(p)
  # w <- get_width(p)
  # print(w)
  # 
  # 
  # localMaxima <- function(x) {
  #   # Use -Inf instead if x is numeric (non-integer)
  #   y <- diff(c(-.Machine$integer.max, x)) > 0L
  #   rle(y)$lengths
  #   y <- cumsum(rle(y)$lengths)
  #   y <- y[seq.int(1L, length(y), 2L)]
  #   if (x[[1]] == x[[2]]) {
  #     y <- y[-1]
  #   }
  #   x[y]
  # }
  # 
  # 
  # a <- as.numeric(gest3$pir0)
  # b <- localMaxima(a)
  # print(b)


