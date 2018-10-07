package com.example.oglescnn;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;

import com.example.eglnn.utils.MessagePackUtils;

import java.io.IOException;

public class TestUtils {

    public static float[][][] getTestCifarImage(Context context) {
        float[][][] inputBatch = new float[3][32][32];
        try {
            float[][][] mean = (float[][][]) MessagePackUtils.unpackParam(context, "cifar10/mean", float[][][].class);
            Bitmap bmp = BitmapFactory.decodeResource(context.getResources(), R.drawable.dog);
            Bitmap bmp2 = Bitmap.createScaledBitmap(bmp, 32, 32, false);
            for (int j = 0; j < 32; ++j) {
                for (int k = 0; k < 32; ++k) {
                    int color = bmp2.getPixel(j, k);
                    inputBatch[0][k][j] = (float) (android.graphics.Color.blue(color)) - mean[0][j][k];
                    inputBatch[1][k][j] = (float) (android.graphics.Color.blue(color)) - mean[1][j][k];
                    inputBatch[2][k][j] = (float) (android.graphics.Color.blue(color)) - mean[2][j][k];
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        return inputBatch;
    }

    public static float[][][] getTestImage(Context context) {
        float[][][] inputBatch = null;
        try {
            inputBatch = (float[][][]) MessagePackUtils.unpackParam(context, "in", float[][][].class);
        } catch (IOException e) {
            e.printStackTrace();
        }
        return inputBatch;
    }

    public static float[][][] getTestImage(Context context, int width, int height) {
        BitmapFactory.Options bfoOptions = new BitmapFactory.Options();
        bfoOptions.inScaled = false;
        Bitmap bmp = BitmapFactory.decodeResource(context.getResources(), R.drawable.cat217);
        Bitmap bmp2 = Bitmap.createScaledBitmap(bmp, width, height, false);
        float[][][] inputBatch = new float[3][height][width];

        for (int j = 0; j < width; ++j) {
            for (int k = 0; k < height; ++k) {
                int color = bmp2.getPixel(j, k);
                inputBatch[0][k][j] = (float) (android.graphics.Color.blue(color)) - 103.939f;
                inputBatch[1][k][j] = (float) (android.graphics.Color.green(color)) - 116.77899f;
                inputBatch[2][k][j] = (float) (android.graphics.Color.red(color)) - 123.68f;
            }
        }
        return inputBatch;
    }

}
