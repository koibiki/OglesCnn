package com.example.oglescnn;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Environment;

import com.example.cnnlib.utils.ParamUnpacker;

public class TestUtils {

    private int mWidthPixel = 32;
    private int mHeightPixel = 32;
    private final String sRoot = Environment.getExternalStorageDirectory().getAbsolutePath();
    private final float[][][] mean;


    public TestUtils() {
        ParamUnpacker pu = new ParamUnpacker();
        mean = (float[][][]) pu.unpackerFunction(sRoot + "/Cifar10es/net/mean.msg", float[][][].class);
    }


    public float[][][] getTestImage(Context context) {

        float[][][] inputBatch = new float[3][mWidthPixel][mHeightPixel];


        // 读取图像源文件 并缩放到预定大小（32）
        Bitmap bmp = BitmapFactory.decodeResource(context.getResources(), R.drawable.airplane);

        Bitmap bmp2 = Bitmap.createScaledBitmap(bmp, mWidthPixel, mHeightPixel, false);


        long begin = System.currentTimeMillis();

        for (int j = 0; j < mWidthPixel; ++j) {
            for (int k = 0; k < mHeightPixel; ++k) {
                // 读取图像的pixel值，并归一化 means矩阵由网络初始化时读取mean.msg生成
                // 注意生成的矩阵顺序为 [batch, channel, width, height]
                int color = bmp2.getPixel(j, k);
                inputBatch[0][k][j] = (float) (android.graphics.Color.blue(color)) - mean[0][j][k];
                inputBatch[1][k][j] = (float) (android.graphics.Color.blue(color)) - mean[1][j][k];
                inputBatch[2][k][j] = (float) (android.graphics.Color.blue(color)) - mean[2][j][k];
            }
        }
        return inputBatch;
    }

}
