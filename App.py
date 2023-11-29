import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import time
import datetime as dt
from datetime import datetime
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
from tensorflow import keras
from keras import layers
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, LSTM, Input, Concatenate, Add
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam

st.set_page_config(
    page_title="Skripsi Angga",
    page_icon=":bar_chart:"
)

with st.sidebar:
    selected = option_menu("Homepage", ["Introduction","Data Preparation", 'Proses Pengujian', 'Prediksi Kedepan'], 
                           icons=['person-hearts','database-fill-gear', 'clipboard-data','graph-up-arrow'], menu_icon="house", default_index=0)

if (selected == "Introduction"):
    st.write("# Hi! Selamat datang di Sistem Prediksi Stock Closing Price 👋🏼")

    st.markdown(
    """
    Project ini dibangun oleh **Nanda Prabu Angganata** yang di bimbing langsung oleh
    ibu **Dr. Eka Mala Sari Rochman, S.Kom., M.Kom.** dan **Ibu Sri Herawati S.Kom., M.Kom.**

    ### Apa saja yang perlu dipersiapkan?
    - Unduh data saham BBCA melalui [yahoo finance](https://finance.yahoo.com/quote/BBCA.JK/history?period1=1356998400&period2=1672531200&interval=1d&filter=history&frequency=1d&includeAdjustedClose=true)
    - Unduh data Nilai Tukar melalui [yahoo finance](https://finance.yahoo.com/quote/IDR%3DX/history?period1=1325376000&period2=1327968000&interval=1d&filter=history&frequency=1d&includeAdjustedClose=true)
    """
    )

if (selected == "Data Preparation"):

    st.write("# Data Preparation! 💾")
    try:
        uploaded_fileSaham = st.file_uploader("Masukkan Data Saham")
        if uploaded_fileSaham is not None:
            dataframeSaham = pd.read_csv(uploaded_fileSaham)
            st.write("Jumlah row pada data `Saham` adalah", dataframeSaham.shape[0])
            st.write("Jumlah column pada data `Saham` adalah", dataframeSaham.shape[1])
            st.table(dataframeSaham.head(5))
            uploaded_fileSaham = True
        else:
            st.info('Data Belum di Inputkan', icon="ℹ️")
    except:
        st.error('Pastikan data anda berekstensi `.CSV`', icon="❗")
        uploaded_fileSaham = False
    
    try:
        uploaded_fileKurs = st.file_uploader("Masukkan Data Nilai Tukar")
        if uploaded_fileKurs is not None:
            dataframeKurs = pd.read_csv(uploaded_fileKurs)
            st.write("Jumlah row pada data `Nilai Tukar` adalah", dataframeKurs.shape[0])
            st.write("Jumlah column pada data `Nilai Tukar` adalah", dataframeKurs.shape[1])
            st.table(dataframeKurs.head(5))
            uploaded_fileKurs = True
        
        else:
            st.info('Data Belum di Inputkan', icon="ℹ️") 
    except:
        st.error('Pastikan data anda berekstensi `.CSV`', icon="❗")
        uploaded_fileKurs = False
    
     
    if uploaded_fileKurs and uploaded_fileSaham is not None:
        try:
            if uploaded_fileKurs or uploaded_fileSaham == True :
                SeriesSaham = st.multiselect(
                "Pilih Kolom Data Saham yang akan digunakan", list(dataframeSaham.columns))
                if not SeriesSaham:
                    st.error("Pilih salah satu Kolom data Saham yang akan digunakan.")
                elif len(SeriesSaham) > 5:
                    st.info('Hanya dapat memilih kolom `Date` dan 4 kolom lain', icon="ℹ️")
                else:
                    st.write("Kolom Data Saham yang digunakan yaitu", SeriesSaham)
    
                SeriesKurs = st.multiselect(
                    "Pilih Kolom Data Kurs yang akan digunakan", list(dataframeKurs.columns)
                    )
                if not SeriesKurs:
                    st.error("Pilih salah satu Kolom data Kurs yang akan digunakan.")
                elif len(SeriesKurs) > 2 or 'Close' not in SeriesKurs:
                    st.info('Hanya dapat memilih kolom `Date` dan `Close`', icon="ℹ️")
                else:
                    st.write("Kolom Data Kurs yang digunakan yaitu", SeriesKurs)
            else:
                st.error('Pastikan data anda berekstensi `.CSV`', icon="❗")

            gabungkan = st.button("Merger Data")
            if gabungkan:
                if not SeriesKurs or not SeriesSaham :
                    st.error("Pilih Kolom data terlebih dahulu", icon="❗")
                else:
                    dataframeSaham = dataframeSaham[SeriesSaham]
                    dataframeKurs = dataframeKurs[SeriesKurs]
                    dataframeKurs = dataframeKurs.rename(columns={"Close":"Nilai Tukar"})
                    dataFinal = pd.merge(dataframeSaham, dataframeKurs, on='Date', how='left')
                    dataFinal = dataFinal.dropna()
                    st.write("## Data setelah digabungkan!")
                    st.write("Jumlah row pada `Data Final` adalah", dataFinal.shape[0])
                    st.write("Jumlah column pada `Data Final` adalah", dataFinal.shape[1])
                    st.table(dataFinal.head(5))
                    dataFinalbaru = dataFinal[["Date","Open","High","Low","Nilai Tukar","Close"]]
                    corrNew = dataFinalbaru.corr(method="pearson", numeric_only=True)
                    st.write("## Korelasi antar variabel!")
                    st.write(corrNew)
                    dataFinal = dataFinal.to_csv()
                    st.download_button(
                        label="Download data as CSV",
                        data=dataFinal,
                        file_name='DataGabungan.csv',
                        mime='csv',)
        except:
            st.error('Perhatikan kembali input data anda', icon="❗")

################################################################# PROSES PENGUJIAN

if (selected == "Proses Pengujian"):
    st.write("# Proses Pengujian! 📊")
    try:
        uploaded_df = st.file_uploader("Masukkan Dataset")
        if uploaded_df is not None:
            dataframe = pd.read_csv(uploaded_df, index_col='Date', parse_dates=True)
            dataframe = dataframe[['Close','High','Open', 'Low','Nilai Tukar']]
            st.write("Jumlah `baris data` adalah", dataframe.shape[0])
            st.write("Jumlah `kolom data` adalah", dataframe.shape[1])
            st.table(dataframe.head(5))

        else:
            st.info('Data Belum di Inputkan', icon="ℹ️")
    
    except:
        st.error('Perhatikan kembali input data anda', icon='❗')

    def plot_dataLSTM(Y_test,Y_hat):
        plt.plot(Y_test,c = 'r')
        plt.plot(Y_hat,c = 'y')
        plt.xlabel('Day')
        plt.ylabel('Price')
        plt.title("BBCA CLOSING PRICE PREDICTION")
        plt.legend(['Actual','Predicted'],loc = 'lower right')
        plt.show()

    st.set_option('deprecation.showPyplotGlobalUse', False)

    def inisiasiTimestepTest(dataTest,timesteps):
        X_test = []
        Y_test = []
        for i in range(timesteps,dataTest.shape[0]):
            X_test.append(dataTest[i - timesteps:i, 1:dataTest.shape[1]])
            Y_test.append(dataTest[i][0])
        X_test,Y_test = np.array(X_test),np.array(Y_test)
        return X_test, Y_test

    def evaluate_model(model):
        Y_hat = model.predict(X_test)
        mse = mean_squared_error(Y_test,Y_hat)
        rmse = math.sqrt(mse)
        mape = mean_absolute_percentage_error(Y_test, Y_hat)
        r = r2_score(Y_test,Y_hat)
        return mse, rmse, mape, r, Y_test, X_test, Y_hat
    
    try:
        if all(col in dataframe.columns for col in ['Open','High','Low','Close','Nilai Tukar']):
            scaler = MinMaxScaler(feature_range=(0,1))
            data_scaled = scaler.fit_transform(dataframe)
            st.write("## Normalisasi Data")
            st.write(data_scaled)

            st.write("## Data Splitting")
            persentase = 80
            dataCTrain = data_scaled[0:int(len(data_scaled)*(persentase/100))]
            dataTest = data_scaled[int(len(data_scaled)*(persentase/100)):]
            dataTrain = dataCTrain[0:int(len(dataCTrain)*0.9)]
            dataVal = dataCTrain[int(len(dataCTrain)*0.9):]
           
            col1, col2, col3 = st.columns([4,5,4])
            col1.write("Total Data Train `{}` samples".format(len(dataTrain)))
            col2.write("Total Data Validation `{}` samples".format(len(dataVal)))
            col3.write("Total Data Test `{}` samples".format(len(dataTest)))
                        
            st.write("## Inisialisasi Timesteps")
            timesteps = st.selectbox('Pilih Timesteps yang akan digunakan',(5, 10, 20, 30))
            st.write('Timesteps yang digunakan yaitu : ', timesteps)

            if timesteps is not None:
                pred = st.button("Mulai Prediksi")
                if pred:
                    X_test,Y_test=inisiasiTimestepTest(dataTest,timesteps)
    
                    if timesteps == 5:
                        modelLSTM = tf.keras.saving.load_model("model/Timesteps5.h5")
                    elif timesteps == 10:
                        modelLSTM = tf.keras.saving.load_model("model/Timesteps10.h5")
                    elif timesteps == 20:
                        modelLSTM = tf.keras.saving.load_model("model/Timesteps20.h5")
                    else:
                        modelLSTM = tf.keras.saving.load_model("model/Timesteps30.h5")

                    mse, rmse, mape, r2_value, asli, data_full, predicted = evaluate_model(modelLSTM)
                    st.write("## Hasil Prediksi Data Testing")
                    st.write('MSE = {}'.format(mse))
                    # st.write('RMSE = {}'.format(rmse))
                    # st.write('MAPE = {}'.format(mape))
                    # st.write('R-Squared Score = {}'.format(r2_value))
                    st.pyplot(plot_dataLSTM(Y_test,predicted))
    
                    prediction_copies_array = np.repeat(predicted,5, axis=-1)
                    pred=scaler.inverse_transform(np.reshape(prediction_copies_array,(len(predicted),5)))[:,0]
                    df_final=dataframe[predicted.shape[0]*-1:]
                    df_final["Close Predictions"] = pred
                    st.write("# Hasil Denormalisasi")
                    st.table(df_final.tail(5))
                    dataFinal = df_final.to_csv()
                    st.download_button(
                        label="Download data as CSV",
                        data=dataFinal,
                        file_name='HasilPengujian.csv',
                        mime='csv',)
            else:
                st.error('Harap pilih timesteps yang digunakan', icon="❗")

        else:
            st.error('Dataset Harus memiliki kolom `Date`, `Close`, `Open`, `Low`, `High` dan `Nilai Tukar`', icon="❗")
    
    except:
        " "    
            
    
################################################### PREDIKSI KEDEPAN

if (selected == "Prediksi Kedepan"):

    st.write("# Prediksi Kedepan! 📈")
    try:
        uploaded_InputNext = st.file_uploader('Masukkan Data Test yang digunakan')
        if uploaded_InputNext is not None:
            seriesNext = pd.read_csv(uploaded_InputNext, index_col='Date', parse_dates=True)
            if all(col in seriesNext.columns for col in ['Open','High','Low','Close','Nilai Tukar']):
                SeriesClear = seriesNext[['Close','High','Open', 'Low','Nilai Tukar']]
                st.write("Jumlah `baris data` adalah", SeriesClear.shape[0])
                st.write("Jumlah `kolom data` adalah", SeriesClear.shape[1])
                st.table(SeriesClear.head(5))
            
            else:
                st.error('Dataset Harus memiliki kolom `Date`, `Close`, `Open`, `Low`, `High` dan `Nilai Tukar`', icon="❗")
         
        else:
            st.info('Data Belum di Inputkan', icon="ℹ️")
    except:
        st.error('Perhatikan kembali input data anda', icon='❗')
               
    timestepsNext = st.selectbox('Pilih Timesteps yang akan digunakan',(5, 10, 20, 30))
    st.write('Timesteps yang digunakan yaitu : ', timestepsNext)

    n_future = st.slider('Pilih Banyaknya hari yang ingin diprediksi', 0, 30)
    st.write("Banyaknya hari yang akan diprediksi yaitu : ",n_future, 'hari')
    if n_future <= 0 :
        st.error("Banyak hari yang akan diprediksi tidak boleh 0")

    MulaiPredict = st.button('Mulai Prediksi')
    if MulaiPredict:
        try:
            dataTraining = pd.read_csv("Dataset/DataGabungan.csv", index_col='Date', parse_dates=True)
            dataTraining = dataTraining[['Close','High','Open', 'Low','Nilai Tukar']]
            Gabungan = pd.concat([dataTraining, SeriesClear])

            #Normalisasi Data
            scalerNext = MinMaxScaler(feature_range=(0,1))
            skalaData = scalerNext.fit_transform(Gabungan)
            data_scaledNext = skalaData[-len(SeriesClear):]
        
            #Inialisasi Timesteps
            def inisiasiTimestepTest(dataTest,timesteps):
                X_testN = []
                Y_testN = []
             
                for i in range(timesteps,dataTest.shape[0]):
                    X_testN.append(dataTest[i - timesteps:i, 1:dataTest.shape[1]])
                    Y_testN.append(dataTest[i][0])
                X_testN,Y_testN = np.array(X_testN),np.array(Y_testN)
                return X_testN, Y_testN
            X_testN,Y_testN=inisiasiTimestepTest(data_scaledNext,timestepsNext)

        
            #Load ModelLSTM
            if timestepsNext == 5:
                modelLSTM = tf.keras.saving.load_model("model/Timesteps5.h5")
            elif timestepsNext == 10:
                modelLSTM = tf.keras.saving.load_model("model/Timesteps10.h5")
            elif timestepsNext == 20:
                modelLSTM = tf.keras.saving.load_model("model/Timesteps20.h5")
            else:
                modelLSTM = tf.keras.saving.load_model("model/Timesteps30.h5")

            #Prediksi Data Test
            PrediksiTest = modelLSTM.predict(X_testN)
            mseNext = mean_squared_error(Y_testN, PrediksiTest)

            #Prediksi Future
            features = X_testN.shape[2]
            predicted_values = []
        
            input_data = data_scaledNext[:,1:][-timestepsNext:]
            for i in range(n_future):
                next_period = modelLSTM.predict(input_data[-timestepsNext:].reshape(1, timestepsNext, features))
                predicted_values.append(next_period)
                copies = np.repeat(next_period, features, axis=1)[-features:]
                input_data = np.append(input_data, copies, axis=0)
            predicted_values = np.array(predicted_values)

            #Akses Tanggal untuk future
            datelist_test = list(seriesNext.index)
            datelist_future = pd.date_range(datelist_test[-1].date(), periods=n_future, freq='1d').tolist()
        
            #Denormalisasi
            Prediction_FutureCopies = np.repeat(predicted_values,5, axis=-1)
            PredFuture = scalerNext.inverse_transform(np.reshape(Prediction_FutureCopies,(len(predicted_values),5)))[:,0]
            Prediction_TestCopies = np.repeat(PrediksiTest,5, axis=-1)
            PredTest = scalerNext.inverse_transform(np.reshape(Prediction_TestCopies,(len(PrediksiTest),5)))[:,0]

            #Rubah ke Dataframe
            Predict_Future = pd.DataFrame(PredFuture, columns=['Next Period']).set_index(pd.Series(datelist_future))
            Predict_Test = pd.DataFrame(PredTest, columns=['Close Predictions']).set_index(pd.Series(datelist_test[timestepsNext:]))
            Data_Asli = pd.DataFrame(SeriesClear[-len(seriesNext) + timestepsNext:], columns=['Close']).set_index(pd.Series(datelist_test[timestepsNext:]))

            #Visualisasi
            st.write("## Hasil Prediksi Kedepan")
            st.write("Nilai Mean Squared Error `{}`".format(mseNext))           
            plt.plot(Data_Asli, c = 'r')
            plt.plot(Predict_Test, c = 'y')
            plt.plot(Predict_Future, c = 'b')
            plt.axvline(x = max(Predict_Test.index), c = 'g', linewidth=2, linestyle='--')
            plt.xlabel('Day')
            plt.ylabel('Price')
            plt.xticks(rotation=45)
            plt.grid(True)
            plt.title("BBCA CLOSING PRICE PREDICTION")
            plt.legend(['Actual','Predicted','Next Period'],loc = 'lower right')
            st.pyplot(plt.show())

            #Visualisasi Tabel
            Predict_FutureNoIN = pd.DataFrame(PredFuture, columns=['Prediksi Kedepan'])
            n_future += 1
            periode = []
            for i in range(1,n_future):
                periode.append(f'ke-{i}')
            periode = np.array(periode)
            periode_Future = pd.DataFrame(periode, columns=['Periode'])
            result_Future = pd.concat([periode_Future, Predict_FutureNoIN], axis=1)
            st.table(result_Future)

            #Download Data
            dataResult = result_Future.to_csv()

            st.download_button(
                label="Download data as CSV",
                data=dataResult,
                file_name='HasilPrediksiKedepan.csv',
                mime='csv')
            
        except:
            st.error("Proses tidak dapat dijalankan, Periksa kembali data anda!")
