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
            Bitmap bmp = BitmapFactory.decodeResource(context.getResources(), R.drawable.airplane);
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

    public static float[][][] getTestImage(Context context, int width, int height) {
        float[][][] inputBatch = new float[3][height][width];
        Bitmap bmp = BitmapFactory.decodeResource(context.getResources(), R.drawable.cat);
        Bitmap bmp2 = Bitmap.createScaledBitmap(bmp, width, height, false);
        for (int j = 0; j < 32; ++j) {
            for (int k = 0; k < 32; ++k) {
                int color = bmp2.getPixel(j, k);
                inputBatch[0][k][j] = (float) (android.graphics.Color.blue(color));
                inputBatch[1][k][j] = (float) (android.graphics.Color.blue(color));
                inputBatch[2][k][j] = (float) (android.graphics.Color.blue(color));
            }
        }
        return inputBatch;
    }

}
