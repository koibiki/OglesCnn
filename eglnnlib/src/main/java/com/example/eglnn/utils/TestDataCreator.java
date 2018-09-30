package com.example.eglnn.utils;

public class TestDataCreator {

    /**
     * 测试用输入数据
     */
    public static float[][][] createInputBuffer(int[] shape, int bias) {
        int channel = shape[2];
        int width = shape[0];
        int height = shape[1];
        float input[][][] = new float[channel][height][width];
        for (int c = 0; c < channel; c++) {
            for (int w = 0; w < width; w++) {
                for (int h = 0; h < height; h++) {
                    input[c][h][w] = w;
                }
            }
        }
        return input;
    }

    /**
     * 测试用 conv kennel
     */
    public static float[][][] createConvKennels(int[] mKennelShape, int mKennelAmount) {
        int k_w = mKennelShape[0];
        int k_h = mKennelShape[1];
        int k_c = mKennelShape[2];
        int align_c = Utils.alignBy4(k_c);
        float[][][] kennenls = new float[mKennelAmount][align_c / 4][(k_w * k_h + 1) * 4];
        for (int a = 0; a < mKennelAmount; a++) {
            for (int w = 0; w < k_w; w++) {
                for (int h = 0; h < k_h; h++) {
                    for (int c = 0; c < k_c; c++) {
                        kennenls[a][c / 4][(w + h * k_w) * 4 + c % 4] = c;
                    }
                }
            }
            kennenls[a][0][k_w * k_h * 4] = 0;       // bias
        }
        return kennenls;
    }

    /**
     * 测试用 conv kennel Winograd 专用
     */
    public static float[][][][] createConvKennels2(int[] mKennelShape, int mKennelAmount) {
        int k_w = mKennelShape[0];
        int k_h = mKennelShape[1];
        int k_c = mKennelShape[2];
        float[][][][] kennenls = new float[mKennelAmount][k_c][k_h][k_w];
        for (int a = 0; a < mKennelAmount; a++) {
            for (int w = 0; w < k_w; w++) {
                for (int h = 0; h < k_h; h++) {
                    for (int c = 0; c < k_c; c++) {
                        kennenls[a][c][h][w] = 1;
                    }
                }
            }
        }
        return kennenls;
    }

    /**
     * 测试用 conv kennel bias
     * */


    /**
     * 测试用全连接 kennel
     */
    public static float[] createFullConnKennel(int alignSize, int[] inputShape, int index) {
        float[] kennel = new float[alignSize];       // 最后一位是bias
        for (int i = 0; i < inputShape[0] * inputShape[1]; i++) {
            for (int j = 0; j < inputShape[2]; j++) {
                kennel[inputShape[2] * i + j] = j;
            }
        }
        kennel[alignSize - 4] = 1;
        return kennel;
    }

}
