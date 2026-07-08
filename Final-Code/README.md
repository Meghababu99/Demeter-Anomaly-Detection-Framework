# FINAL-CODE

1. **Convert and prepare raw DEMETER data**
   - `IMSC_ICE_ASCII.ipynb`
   - `GRID-data-Preperation.py`

2. **Inspect gridded data and descriptive statistics**
   - `Grid-Timesiers-plot.ipynb`
   - `Data_statistics.ipynb`
   - `EQ_statistics.ipynb`
   - `Heatmap-correlation.ipynb`
   - `Q_FB-CorrelationPlots-GRID.ipynb`

3. **Create rolling-window background-corrected data**
   - `Rolling_Window_BGcorrected_Data.ipynb`
   - `Rolling_Window-BGcorrected-Data.py`
   - `BackgroundCorrection-Plot.ipynb`

4. **Train and evaluate LSTM autoencoder models**
   - `DEMETER_LSTM_AE-Training.ipynb`
   - `DEMETER_LSTM_AE-Training-loadpretrained.ipynb`
   - `Retraining_Hp-B.sh`

5. **Prepare labels, random baseline, and threshold tests**
   - `Seismic-Labelling_Summary--RDM_prep.ipynb`
   - `Random_baseline_rolling_windows.ipynb` ### Right way


6. **Summarise final results**
   - `Results_PLOT_SW22TW48.ipynb`
   - `Best-Performing-Criteria.ipynb`
   - `SW22TW48_leadtime.ipynb`
   - `Anomaly_threshold_analysis.ipynb`




## Notes for interpretation

This code supports a statistical association study. The model outputs are reconstruction-error anomalies, not direct earthquake predictions. Any result should be reported with the random baseline, uncertainty/error bars, threshold choice, spatial window, temporal window, and storm-correction status.
