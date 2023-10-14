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
        
        uploaded_fileKurs = st.file_uploader("Masukkan Data Nilai Tukar")
        if uploaded_fileKurs is not None:
            dataframeKurs = pd.read_csv(uploaded_fileKurs)
            st.write("Jumlah row pada data `Nilai Tukar` adalah", dataframeKurs.shape[0])
            st.write("Jumlah column pada data `Nilai Tukar` adalah", dataframeKurs.shape[1])
            st.table(dataframeKurs.head(5))

    except:
        st.info('Data Belum di Inputkan', icon="ℹ️")
    
    try:
        SeriesSaham = st.multiselect(
            "Pilih Kolom Data Saham yang akan digunakan", list(dataframeSaham.columns)
            )
        if not SeriesSaham:
            st.error("Pilih salah satu Kolom data Saham yang akan digunakan.")
        else:
            st.write("Kolom Data Saham yang digunakan yaitu", SeriesSaham)
    
        SeriesKurs = st.multiselect(
            "Pilih Kolom Data Kurs yang akan digunakan", list(dataframeKurs.columns)
            )
        if not SeriesKurs:
            st.error("Pilih salah satu Kolom data Kurs yang akan digunakan.")
        else:
            st.write("Kolom Data Kurs yang digunakan yaitu", SeriesKurs)

    except:
        st.info('Data Belum di Inputkan', icon="ℹ️")


    if uploaded_fileKurs is not None:
        try:
            gabungkan = st.button("Merger Data")
            if gabungkan:
                if not SeriesKurs or not SeriesSaham :
                    st.error("Pilih Kolom data terlebih dahulu")
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
            "ada yang error nich"

################################################################# PROSES PENGUJIAN

if (selected == "Proses Pengujian"):
    st.write("# Proses Pengujian! 📊")
    uploaded_df = st.file_uploader("Masukkan Dataset")
    if uploaded_df is not None:
        dataframe = pd.read_csv(uploaded_df, index_col='Date', parse_dates=True)
        dataframe = dataframe[['Close','High','Open', 'Low','Nilai Tukar']]
        st.write("Jumlah `baris data` adalah", dataframe.shape[0])
        st.write("Jumlah `kolom data` adalah", dataframe.shape[1])
        st.table(dataframe.head(5))
    
    else:
        st.info('Data Belum di Inputkan', icon="ℹ️")

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
    scaler = MinMaxScaler(feature_range=(0,1))
    data_scaled = scaler.fit_transform(dataframe)
    st.write("## Normalisasi Data")
    st.write(data_scaled)

    st.write("## Data Splitting")
    option = st.selectbox('Persentase Data untuk Proses Training',(70, 80, 90))
    st.write('Persentase Data Train :', option)

    dataCTrain = data_scaled[0:int(len(data_scaled)*(option/100))]
    dataTest = data_scaled[int(len(data_scaled)*(option/100)):]
    dataTrain = dataCTrain[0:int(len(dataCTrain)*0.9)]
    dataVal = dataCTrain[int(len(dataCTrain)*0.9):]
    st.write("Total Data Train {} samples".format(len(dataTrain)))
    st.write("Total Data Validation {} samples".format(len(dataVal)))
    st.write("Total Data Test {} samples".format(len(dataTest)))

    earlystop = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=80,  verbose=1, mode='min')
    callbacks_list = [earlystop]

    timesteps = st.selectbox('Pilih Timesteps yang akan digunakan',(5, 10, 20, 30))
    st.write('Timesteps yang digunakan yaitu : ', timesteps)

except:
    " "    

try:
    if timesteps is not None:
        pred = st.button("Mulai Prediksi")
        if pred:
            X_test,Y_test=inisiasiTimestepTest(dataTest,timesteps)
    
            if timesteps == 5:
                modelLSTM = tf.keras.saving.load_model(r"D:\UNIVERSITAS TRUNOJOYO MADURA\SEMESTER 8\BISSMILAH SKRIPSI\PROGRESS\TOPIK FORECAST\GUI\Model\Timesteps5.h5")
            elif timesteps == 10:
                modelLSTM = tf.keras.saving.load_model(r"D:\UNIVERSITAS TRUNOJOYO MADURA\SEMESTER 8\BISSMILAH SKRIPSI\PROGRESS\TOPIK FORECAST\GUI\Model\Timesteps10.h5")
            elif timesteps == 20:
                modelLSTM = tf.keras.saving.load_model(r"D:\UNIVERSITAS TRUNOJOYO MADURA\SEMESTER 8\BISSMILAH SKRIPSI\PROGRESS\TOPIK FORECAST\GUI\Model\Timesteps20.h5")
            else:
                modelLSTM = tf.keras.saving.load_model(r"D:\UNIVERSITAS TRUNOJOYO MADURA\SEMESTER 8\BISSMILAH SKRIPSI\PROGRESS\TOPIK FORECAST\GUI\Model\Timesteps30.h5")

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
                file_name='DataGabungan.csv',
                mime='csv',)

except:
    " "

################################################### PREDIKSI KEDEPAN

if (selected == "Prediksi Kedepan"):

    st.write("# Prediksi Kedepan! 📈")

    pilihDataTest = st.toggle("Gunakan Data Test Default")

    if pilihDataTest:
        st.write("Anda akan menggunakan Data Test Default")
        seriesNext = pd.read_csv(r"D:\UNIVERSITAS TRUNOJOYO MADURA\SEMESTER 8\BISSMILAH SKRIPSI\PROGRESS\TOPIK FORECAST\GUI\DataGabungan.csv", index_col='Date', parse_dates=True)
        SeriesClear = seriesNext[['Close','High','Open', 'Low','Nilai Tukar']]
        
    else:
        uploaded_InputNext = st.file_uploader('Masukkan Data Test yang digunakan')
        if uploaded_InputNext is not None:
            seriesNext = pd.read_csv(uploaded_InputNext, index_col='Date', parse_dates=True)
            SeriesClear = seriesNext[['Close','High','Open', 'Low','Nilai Tukar']]
            st.write("Jumlah `baris data` adalah", SeriesClear.shape[0])
            st.write("Jumlah `kolom data` adalah", SeriesClear.shape[1])
            st.table(SeriesClear.head(5))

        else:
            st.info('Data Belum di Inputkan', icon="ℹ️")


    uploaded_NextPeriod = st.file_uploader('Masukkan Data untuk Memprediksi periode Kedepan')
    if uploaded_NextPeriod is not None:
        df_2023 = pd.read_csv(uploaded_NextPeriod, index_col='Date', parse_dates=True)
        series2023 = df_2023[['High','Open', 'Low','Nilai Tukar']]
        st.write("Jumlah `baris data` adalah", series2023.shape[0])
        st.write("Jumlah `kolom data` adalah", series2023.shape[1])
        st.table(series2023.head(5))

    else:
        st.info('Data Belum di Inputkan', icon="ℹ️")
    
    timestepsNext = st.selectbox('Pilih Timesteps yang akan digunakan',(5, 10, 20, 30))
    st.write('Timesteps yang digunakan yaitu : ', timestepsNext)

    n_future = st.slider('Pilih Banyaknya hari yang ingin diprediksi', 0, 30)
    st.write("Banyaknya hari yang akan diprediksi yaitu : ",n_future, 'hari')
    if n_future <= 0 :
        st.error("Banyak hari yang akan diprediksi tidak boleh 0")

    MulaiPredict = st.button('Mulai Prediksi')
    if MulaiPredict:
        series23 = series2023[:n_future]
        gabung = pd.concat([SeriesClear, series23])

        #NormalisasiData
        scalerNext = MinMaxScaler(feature_range=(0,1))
        skalaData = scalerNext.fit_transform(gabung)
        sizeDataAsli = len(gabung) - len(series23)
        data_scaledNext = skalaData[:sizeDataAsli]

        #Split Data
        dataCTrainNext = data_scaledNext[0:int(len(data_scaledNext)*0.8)]
        dataTestNext = data_scaledNext[int(len(data_scaledNext)*0.8):]
        dataTrainNext = dataCTrainNext[0:int(len(dataCTrainNext)*0.9)]
        dataValNext = dataCTrainNext[int(len(dataCTrainNext)*0.9):]

        dataFuture = skalaData[-n_future:]
        result = np.concatenate((dataTestNext, dataFuture))


        #Inisialisasi Timesteps
        def inisiasiTimestepTest(dataTest,timesteps):
            X_testN = []
            Y_testN = []
             
            for i in range(timesteps,dataTest.shape[0]):
                X_testN.append(dataTest[i - timesteps:i, 1:dataTest.shape[1]])
                Y_testN.append(dataTest[i][0])
            X_testN,Y_testN = np.array(X_testN),np.array(Y_testN)
            return X_testN, Y_testN
        
        Data,Y_testN=inisiasiTimestepTest(result,timestepsNext)
        untukdatatest = len(result) - n_future
        X_Future = Data[-n_future:]
        X_testN = Data[:untukdatatest - timestepsNext]
        Y_testN = Y_testN[:untukdatatest - timestepsNext]


        #Load ModelLSTM
        if timestepsNext == 5:
            modelLSTM = tf.keras.saving.load_model(r"D:\UNIVERSITAS TRUNOJOYO MADURA\SEMESTER 8\BISSMILAH SKRIPSI\PROGRESS\TOPIK FORECAST\GUI\Model\Timesteps5.h5")
        elif timestepsNext == 10:
            modelLSTM = tf.keras.saving.load_model(r"D:\UNIVERSITAS TRUNOJOYO MADURA\SEMESTER 8\BISSMILAH SKRIPSI\PROGRESS\TOPIK FORECAST\GUI\Model\Timesteps10.h5")
        elif timestepsNext == 20:
            modelLSTM = tf.keras.saving.load_model(r"D:\UNIVERSITAS TRUNOJOYO MADURA\SEMESTER 8\BISSMILAH SKRIPSI\PROGRESS\TOPIK FORECAST\GUI\Model\Timesteps20.h5")
        else:
            modelLSTM = tf.keras.saving.load_model(r"D:\UNIVERSITAS TRUNOJOYO MADURA\SEMESTER 8\BISSMILAH SKRIPSI\PROGRESS\TOPIK FORECAST\GUI\Model\Timesteps30.h5")


        #Prediksi Data Test
        PrediksiTest = modelLSTM.predict(X_testN)
        mse = mean_squared_error(Y_testN, PrediksiTest)
        rmse = math.sqrt(mse)
        r = r2_score(Y_testN,PrediksiTest)
        mape = mean_absolute_percentage_error(Y_testN,PrediksiTest)

        #Prediksi Next period
        FuturePrediction = modelLSTM.predict(X_Future)

        #Ambil Date
        DateTest = SeriesClear.index[-len(dataTestNext) + timestepsNext:]
        DateNextPeriod = series2023.index[:n_future]
          
        #Denormalisasi
        NextPeriodCopies = np.repeat(FuturePrediction,5, axis=-1)
        NextPeriodDenormal = scalerNext.inverse_transform(np.reshape(NextPeriodCopies,(len(FuturePrediction),5)))[:,0]

        TestPredCopies = np.repeat(PrediksiTest,5, axis=-1)
        TestPredDenorm = scalerNext.inverse_transform(np.reshape(TestPredCopies,(len(PrediksiTest),5)))[:,0]

        Asli = SeriesClear['Close'][-len(dataTestNext) + timestepsNext:]

        HasilFuturePredict = pd.DataFrame(NextPeriodDenormal, columns=['Next Period']).set_index(pd.Series(DateNextPeriod[:n_future]))
        TestPrediction = pd.DataFrame(TestPredDenorm, columns=['Close Predictions']).set_index(pd.Series(DateTest))
        DataAsli = pd.DataFrame(Asli, columns=['Close']).set_index(pd.Series(DateTest))

        #Visualisasi
        st.write("# Hasil Prediksi")
        plt.plot(DataAsli, c = 'r')
        plt.plot(TestPrediction, c = 'y')
        plt.plot(HasilFuturePredict, c = 'b')
        plt.axvline(x = max(seriesNext.index), c = 'g', linewidth=2, linestyle='--')
        plt.xlabel('Day')
        plt.ylabel('Price')
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.title("BBCA CLOSING PRICE PREDICTION")
        plt.legend(['Actual','Predicted','Next Period'],loc = 'lower right')
        fig = plt.show()
        st.pyplot(fig)

            # FuturePredictionDenormal = pd.DataFrame(NextPeriod_Denormal, columns=['Hasil Prediksi Kedepan'])
            # df_NextPeriod = df_NextPeriod.reset_index()
            # day = df_NextPeriod['Date'][0:n_future]
            # tanggal = pd.DataFrame(day)
            # HasilNextPeriod = pd.concat([tanggal,FuturePredictionDenormal], axis=1)
        st.table(HasilFuturePredict)
    