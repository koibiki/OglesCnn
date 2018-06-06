package com.example.cnnlib.utils;

import android.os.Environment;
import android.util.Log;

import java.io.Closeable;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;

public class LogUtils {

    private static final String TAG = "LogUtils";
    private static final String sRoot = Environment.getExternalStorageDirectory() + "/0output/";

    public static void printf(float[] array, int width, String name) {
        int len = array.length / 4;
        float[][] arr = new float[4][len];
        for (int i = 0; i < len; i++) {
            arr[0][i] = array[i * 4];
            arr[1][i] = array[i * 4 + 1];
            arr[2][i] = array[i * 4 + 2];
            arr[3][i] = array[i * 4 + 3];
        }
        printToFile(arr, width, name);
    }


    private static void printToTerminal(float[][] array, int width, String name) {
        Log.w(TAG, "-------" + name + "---------------------------------------");
        StringBuffer sb = new StringBuffer();
        for (float[] anArray : array) {
            int len = anArray.length;
            for (int ii = 0; ii < len; ii++) {
                sb.append(anArray[ii]).append(" ,");
                if ((ii + 1) % width == 0) {
                    Log.w(TAG, sb.toString());
                    sb = new StringBuffer();
                }
            }
        }
    }

    private static void printToFile(float[][] array, int width, String name) {
        Log.w(TAG, "-------" + name + "---------------------------------------");
        FileWriter fileWriter = null;
        try {
            String fileName = sRoot + name;
            File file = new File(fileName);
            if (!file.getParentFile().exists()) {
                file.getParentFile().mkdirs();
            }
            fileWriter = new FileWriter(file);

            for (int i=0;i<4;i++) {
                int len = array[i].length;
                for (int ii = 0; ii < len; ii++) {
                    if ((ii + 1) % width == 0) {
                        fileWriter.write("\n");
                        fileWriter.write("第"+i+"层第"+(ii+1)/width+"行");
                    }
                    fileWriter.write(array[i][ii] + " ,");
                }
                fileWriter.write("\n");
                fileWriter.write("\n");
            }
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            close(fileWriter);
        }


    }

    private static void close(Closeable closeable) {
        if (closeable != null) {
            try {
                closeable.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }

}
