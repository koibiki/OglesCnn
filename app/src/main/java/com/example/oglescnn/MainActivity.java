package com.example.oglescnn;

import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.view.View;

import com.example.eglnn.NnNetwork;
import com.example.eglnn.layer.Concat;
import com.example.eglnn.layer.Conv;
import com.example.eglnn.layer.ConvGEMM;
import com.example.eglnn.layer.ConvWinogradF23;
import com.example.eglnn.layer.Expand;
import com.example.eglnn.layer.Layer.PaddingType;
import com.example.eglnn.layer.Input;
import com.example.eglnn.layer.Layer;
import com.example.eglnn.layer.Pooling;
import com.example.eglnn.utils.TestDataCreator;
import com.example.eglnn.utils.Utils;

public class MainActivity extends AppCompatActivity {

    private NnNetwork mNnNetwork;

    int width = 227;
    int height = width;
    int channel = 4;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

//        buildSqueezeNet();
//        buildSqueezeNet2();
//        buildTestNet();

        buildCifar10Net();
    }

    private void buildCifar10Net() {
        mNnNetwork = new NnNetwork(this);

        Layer in = new Input(this, "input", 32, 32, 3);
        mNnNetwork.addLayer(in);

        Layer conv1 = new ConvGEMM(this, "conv1", in, 32, 5, 5, PaddingType.SAME, 1, 1, Layer.ActiveType.RELU);
        mNnNetwork.addLayer(conv1);

        mNnNetwork.initialize();
    }

    private void buildTestNet() {
        mNnNetwork = new NnNetwork(this);

        Layer in = new Input(this, "input", width, height, channel);
        mNnNetwork.addLayer(in);

        Layer conv1 = new ConvGEMM(this, "conv1", in, 4, 3, 3, PaddingType.VALID, 2, 2, Layer.ActiveType.RELU);
        mNnNetwork.addLayer(conv1);

        mNnNetwork.initialize();
    }

    private void buildSqueezeNet() {
        mNnNetwork = new NnNetwork(this);

        Layer in = new Input(this, "input", width, height, channel);
        mNnNetwork.addLayer(in);

        Layer conv1 = new Conv(this, "conv1", in, 64, 3, 3, PaddingType.VALID, 2, 2, Layer.ActiveType.RELU);
        mNnNetwork.addLayer(conv1);

        Pooling pooling1 = new Pooling(this, "pool1", conv1, 3, 3, PaddingType.VALID, 2, 2);
        mNnNetwork.addLayer(pooling1);

        // fire2
        Layer conv2_squeeze = new Conv(this, "conv2_squeeze", pooling1, 16, 1, 1, PaddingType.SAME, 1, 1, Layer.ActiveType.RELU);
        mNnNetwork.addLayer(conv2_squeeze);

        Layer conv2_1 = new Conv(this, "conv2_expand1", conv2_squeeze, 64, 1, 1, PaddingType.SAME, 1, 1, Layer.ActiveType.RELU);
        mNnNetwork.addLayer(conv2_1);

        Layer conv2_2 = new Conv(this, "conv2_expand3", conv2_squeeze, 64, 3, 3, PaddingType.SAME, 1, 1, Layer.ActiveType.RELU);
        mNnNetwork.addLayer(conv2_2);

        Concat concat2 = new Concat(this, "concat2", new Layer[]{conv2_1, conv2_2}, 2);
        mNnNetwork.addLayer(concat2);

        // fire3
        Layer conv3_squeeze = new Conv(this, "conv3_squeeze", concat2, 16, 1, 1, PaddingType.SAME, 1, 1, Layer.ActiveType.RELU);
        mNnNetwork.addLayer(conv3_squeeze);

        Layer conv3_1 = new Conv(this, "conv3_expand1", conv3_squeeze, 64, 1, 1, PaddingType.SAME, 1, 1, Layer.ActiveType.RELU);
        mNnNetwork.addLayer(conv3_1);

        Layer conv3_2 = new Conv(this, "conv3_expand3", conv3_squeeze, 64, 3, 3, PaddingType.SAME, 1, 1, Layer.ActiveType.RELU);
        mNnNetwork.addLayer(conv3_2);

        Concat concat3 = new Concat(this, "concat3", new Layer[]{conv3_1, conv3_2}, 2);
        mNnNetwork.addLayer(concat3);

        Pooling pooling3 = new Pooling(this, "pool3", concat3, 3, 3, PaddingType.VALID, 2, 2);
        mNnNetwork.addLayer(pooling3);

        // fire4
        Layer conv4_squeeze = new Conv(this, "conv4_squeeze", pooling3, 32, 1, 1, PaddingType.SAME, 1, 1, Layer.ActiveType.RELU);
        mNnNetwork.addLayer(conv4_squeeze);

        Layer conv4_1 = new Conv(this, "conv4_expand1", conv4_squeeze, 128, 1, 1, PaddingType.SAME, 1, 1, Layer.ActiveType.RELU);
        mNnNetwork.addLayer(conv4_1);

        Layer conv4_2 = new Conv(this, "conv4_expand3", conv4_squeeze, 128, 3, 3, PaddingType.SAME, 1, 1, Layer.ActiveType.RELU);
        mNnNetwork.addLayer(conv4_2);

        Concat concat4 = new Concat(this, "concat4", new Layer[]{conv4_1, conv4_2}, 2);
        mNnNetwork.addLayer(concat4);

        // fire5
        Layer conv5_squeeze = new Conv(this, "conv5_squeeze", concat4, 32, 1, 1, PaddingType.SAME, 1, 1, Layer.ActiveType.RELU);
        mNnNetwork.addLayer(conv5_squeeze);

        Layer conv5_1 = new Conv(this, "conv5_expand1", conv5_squeeze, 128, 1, 1, PaddingType.SAME, 1, 1, Layer.ActiveType.RELU);
        mNnNetwork.addLayer(conv5_1);

        Layer conv5_2 = new Conv(this, "conv5_expand3", conv5_squeeze, 128, 3, 3, PaddingType.SAME, 1, 1, Layer.ActiveType.RELU);
        mNnNetwork.addLayer(conv5_2);

        Concat concat5 = new Concat(this, "concat5", new Layer[]{conv5_1, conv5_2}, 2);
        mNnNetwork.addLayer(concat5);

        Pooling pooling5 = new Pooling(this, "pool5", concat5, 3, 3, PaddingType.VALID, 2, 2);
        mNnNetwork.addLayer(pooling5);

        // fire6
        Layer conv6_squeeze = new Conv(this, "conv6_squeeze", pooling5, 48, 1, 1, PaddingType.SAME, 1, 1, Layer.ActiveType.RELU);
        mNnNetwork.addLayer(conv6_squeeze);

        Layer conv6_1 = new Conv(this, "conv6_expand1", conv6_squeeze, 192, 1, 1, PaddingType.SAME, 1, 1, Layer.ActiveType.RELU);
        mNnNetwork.addLayer(conv6_1);

        Layer conv6_2 = new Conv(this, "conv6_expand3", conv6_squeeze, 192, 3, 3, PaddingType.SAME, 1, 1, Layer.ActiveType.RELU);
        mNnNetwork.addLayer(conv6_2);

        Concat concat6 = new Concat(this, "concat6", new Layer[]{conv6_1, conv6_2}, 2);
        mNnNetwork.addLayer(concat6);

        // fire7
        Layer conv7_squeeze = new Conv(this, "conv7_squeeze", concat6, 48, 1, 1, PaddingType.SAME, 1, 1, Layer.ActiveType.RELU);
        mNnNetwork.addLayer(conv7_squeeze);

        Layer conv7_1 = new Conv(this, "conv7_expand1", conv7_squeeze, 192, 1, 1, PaddingType.SAME, 1, 1, Layer.ActiveType.RELU);
        mNnNetwork.addLayer(conv7_1);

        Layer conv7_2 = new Conv(this, "conv7_expand3", conv7_squeeze, 192, 3, 3, PaddingType.SAME, 1, 1, Layer.ActiveType.RELU);
        mNnNetwork.addLayer(conv7_2);

        Concat concat7 = new Concat(this, "concat7", new Layer[]{conv7_1, conv7_2}, 2);
        mNnNetwork.addLayer(concat7);

        // fire8
        Layer conv8_squeeze = new Conv(this, "conv8_squeeze", concat7, 64, 1, 1, PaddingType.SAME, 1, 1, Layer.ActiveType.RELU);
        mNnNetwork.addLayer(conv8_squeeze);

        Layer conv8_1 = new Conv(this, "conv8_expand1", conv8_squeeze, 256, 1, 1, PaddingType.SAME, 1, 1, Layer.ActiveType.RELU);
        mNnNetwork.addLayer(conv8_1);

        Layer conv8_2 = new Conv(this, "conv8_expand3", conv8_squeeze, 256, 3, 3, PaddingType.SAME, 1, 1, Layer.ActiveType.RELU);
        mNnNetwork.addLayer(conv8_2);

        Concat concat8 = new Concat(this, "concat8", new Layer[]{conv8_1, conv8_2}, 2);
        mNnNetwork.addLayer(concat8);

        // fire9
        Layer conv9_squeeze = new Conv(this, "conv9_squeeze", concat8, 64, 1, 1, PaddingType.SAME, 1, 1, Layer.ActiveType.RELU);
        mNnNetwork.addLayer(conv9_squeeze);

        Layer conv9_1 = new Conv(this, "conv9_expand1", conv9_squeeze, 256, 1, 1, PaddingType.SAME, 1, 1, Layer.ActiveType.RELU);
        mNnNetwork.addLayer(conv9_1);

        Layer conv9_2 = new Conv(this, "conv9_expand3", conv9_squeeze, 256, 3, 3, PaddingType.SAME, 1, 1, Layer.ActiveType.RELU);
        mNnNetwork.addLayer(conv9_2);

        Concat concat9 = new Concat(this, "concat9", new Layer[]{conv9_1, conv9_2}, 2);
        mNnNetwork.addLayer(concat9);

        // fire10
        Layer conv10 = new Conv(this, "conv10", concat9, 1000, 1, 1, PaddingType.SAME, 1, 1, Layer.ActiveType.RELU);
        mNnNetwork.addLayer(conv10);

//        Pooling pooling11 = new Pooling(this, conv10, 13, 13, PaddingType.VALID, 1, 1);
//        mNnNetwork.addLayer(pooling11);

        mNnNetwork.initialize();
    }

    private void buildSqueezeNet2() {
        mNnNetwork = new NnNetwork(this);

        Layer in = new Input(this, "input", width, height, channel);
        mNnNetwork.addLayer(in);

        Layer conv1 = new ConvGEMM(this, "conv1", in, 64, 3, 3, PaddingType.VALID, 2, 2, Layer.ActiveType.RELU);
        mNnNetwork.addLayer(conv1);

        Pooling pooling1 = new Pooling(this, "pool1", conv1, 3, 3, PaddingType.VALID, 2, 2);
        mNnNetwork.addLayer(pooling1);

        // fire2
        Layer conv2_squeeze = new ConvGEMM(this, "conv2_squeeze", pooling1, 16, 1, 1, PaddingType.SAME, 1, 1, Layer.ActiveType.RELU);
        mNnNetwork.addLayer(conv2_squeeze);

        Layer conv2_1 = new ConvGEMM(this, "conv2_expand1", conv2_squeeze, 64, 1, 1, PaddingType.SAME, 1, 1, Layer.ActiveType.RELU);
        mNnNetwork.addLayer(conv2_1);

        Layer conv2_2 = new ConvGEMM(this, "conv2_expand3", conv2_squeeze, 64, 3, 3, PaddingType.SAME, 1, 1, Layer.ActiveType.RELU);
        mNnNetwork.addLayer(conv2_2);

        Concat concat2 = new Concat(this, "concat2", new Layer[]{conv2_1, conv2_2}, 2);
        mNnNetwork.addLayer(concat2);

        // fire3
        Layer conv3_squeeze = new ConvGEMM(this, "conv3_squeeze", concat2, 16, 1, 1, PaddingType.SAME, 1, 1, Layer.ActiveType.RELU);
        mNnNetwork.addLayer(conv3_squeeze);

        Layer conv3_1 = new ConvGEMM(this, "conv3_expand1", conv3_squeeze, 64, 1, 1, PaddingType.SAME, 1, 1, Layer.ActiveType.RELU);
        mNnNetwork.addLayer(conv3_1);

        Layer conv3_2 = new ConvGEMM(this, "conv3_expand3", conv3_squeeze, 64, 3, 3, PaddingType.SAME, 1, 1, Layer.ActiveType.RELU);
        mNnNetwork.addLayer(conv3_2);

        Concat concat3 = new Concat(this, "concat3", new Layer[]{conv3_1, conv3_2}, 2);
        mNnNetwork.addLayer(concat3);

        Pooling pooling3 = new Pooling(this, "pool3", concat3, 3, 3, PaddingType.VALID, 2, 2);
        mNnNetwork.addLayer(pooling3);

        // fire4
        Layer conv4_squeeze = new ConvGEMM(this, "conv4_squeeze", pooling3, 32, 1, 1, PaddingType.SAME, 1, 1, Layer.ActiveType.RELU);
        mNnNetwork.addLayer(conv4_squeeze);

        Layer conv4_1 = new ConvGEMM(this, "conv4_expand1", conv4_squeeze, 128, 1, 1, PaddingType.SAME, 1, 1, Layer.ActiveType.RELU);
        mNnNetwork.addLayer(conv4_1);

        Layer conv4_2 = new ConvGEMM(this, "conv4_expand3", conv4_squeeze, 128, 3, 3, PaddingType.SAME, 1, 1, Layer.ActiveType.RELU);
        mNnNetwork.addLayer(conv4_2);

        Concat concat4 = new Concat(this, "concat4", new Layer[]{conv4_1, conv4_2}, 2);
        mNnNetwork.addLayer(concat4);

        // fire5
        Layer conv5_squeeze = new ConvGEMM(this, "conv5_squeeze", concat4, 32, 1, 1, PaddingType.SAME, 1, 1, Layer.ActiveType.RELU);
        mNnNetwork.addLayer(conv5_squeeze);

        Layer conv5_1 = new ConvGEMM(this, "conv5_expand1", conv5_squeeze, 128, 1, 1, PaddingType.SAME, 1, 1, Layer.ActiveType.RELU);
        mNnNetwork.addLayer(conv5_1);

        Layer conv5_2 = new ConvGEMM(this, "conv5_expand3", conv5_squeeze, 128, 3, 3, PaddingType.SAME, 1, 1, Layer.ActiveType.RELU);
        mNnNetwork.addLayer(conv5_2);

        Concat concat5 = new Concat(this, "concat5", new Layer[]{conv5_1, conv5_2}, 2);
        mNnNetwork.addLayer(concat5);

        Pooling pooling5 = new Pooling(this, "pool5", concat5, 3, 3, PaddingType.VALID, 2, 2);
        mNnNetwork.addLayer(pooling5);

        // fire6
        Layer conv6_squeeze = new ConvGEMM(this, "conv6_squeeze", pooling5, 48, 1, 1, PaddingType.SAME, 1, 1, Layer.ActiveType.RELU);
        mNnNetwork.addLayer(conv6_squeeze);

        Layer conv6_1 = new ConvGEMM(this, "conv6_expand1", conv6_squeeze, 192, 1, 1, PaddingType.SAME, 1, 1, Layer.ActiveType.RELU);
        mNnNetwork.addLayer(conv6_1);

        Layer conv6_2 = new ConvGEMM(this, "conv6_expand3", conv6_squeeze, 192, 3, 3, PaddingType.SAME, 1, 1, Layer.ActiveType.RELU);
        mNnNetwork.addLayer(conv6_2);

        Concat concat6 = new Concat(this, "concat6", new Layer[]{conv6_1, conv6_2}, 2);
        mNnNetwork.addLayer(concat6);

        // fire7
        Layer conv7_squeeze = new ConvGEMM(this, "conv7_squeeze", concat6, 48, 1, 1, PaddingType.SAME, 1, 1, Layer.ActiveType.RELU);
        mNnNetwork.addLayer(conv7_squeeze);

        Layer conv7_1 = new ConvGEMM(this, "conv7_expand1", conv7_squeeze, 192, 1, 1, PaddingType.SAME, 1, 1, Layer.ActiveType.RELU);
        mNnNetwork.addLayer(conv7_1);

        Layer conv7_2 = new ConvGEMM(this, "conv7_expand3", conv7_squeeze, 192, 3, 3, PaddingType.SAME, 1, 1, Layer.ActiveType.RELU);
        mNnNetwork.addLayer(conv7_2);

        Concat concat7 = new Concat(this, "concat7", new Layer[]{conv7_1, conv7_2}, 2);
        mNnNetwork.addLayer(concat7);

        // fire8
        Layer conv8_squeeze = new ConvGEMM(this, "conv8_squeeze", concat7, 64, 1, 1, PaddingType.SAME, 1, 1, Layer.ActiveType.RELU);
        mNnNetwork.addLayer(conv8_squeeze);

        Layer conv8_1 = new ConvGEMM(this, "conv8_expand1", conv8_squeeze, 256, 1, 1, PaddingType.SAME, 1, 1, Layer.ActiveType.RELU);
        mNnNetwork.addLayer(conv8_1);

        Layer conv8_2 = new ConvGEMM(this, "conv8_expand3", conv8_squeeze, 256, 3, 3, PaddingType.SAME, 1, 1, Layer.ActiveType.RELU);
        mNnNetwork.addLayer(conv8_2);

        Concat concat8 = new Concat(this, "concat8", new Layer[]{conv8_1, conv8_2}, 2);
        mNnNetwork.addLayer(concat8);

        // fire9
        Layer conv9_squeeze = new ConvGEMM(this, "conv9_squeeze", concat8, 64, 1, 1, PaddingType.SAME, 1, 1, Layer.ActiveType.RELU);
        mNnNetwork.addLayer(conv9_squeeze);

        Layer conv9_1 = new ConvGEMM(this, "conv9_expand1", conv9_squeeze, 256, 1, 1, PaddingType.SAME, 1, 1, Layer.ActiveType.RELU);
        mNnNetwork.addLayer(conv9_1);

        Layer conv9_2 = new ConvGEMM(this, "conv9_expand3", conv9_squeeze, 256, 3, 3, PaddingType.SAME, 1, 1, Layer.ActiveType.RELU);
        mNnNetwork.addLayer(conv9_2);

        Concat concat9 = new Concat(this, "concat9", new Layer[]{conv9_1, conv9_2}, 2);
        mNnNetwork.addLayer(concat9);

        // fire10
        Layer conv10 = new ConvGEMM(this, "conv10", concat9, 1000, 1, 1, PaddingType.SAME, 1, 1, Layer.ActiveType.RELU);
        mNnNetwork.addLayer(conv10);

//        Pooling pooling11 = new Pooling(this, conv10, 13, 13, PaddingType.VALID, 1, 1);
//        mNnNetwork.addLayer(pooling11);

        mNnNetwork.initialize();

    }


    public void runNn(View view) {
//        float[][][] input = TestDataCreator.createInputBuffer(new int[]{32, 32, 3}, 0);
        float[][][] input = TestUtils.getTestCifarImage(this);
        float[][] localInput = new float[Utils.alignBy4(3) / 4][32 * 32 * 4];
        for (int w = 0; w < 32; w++) {
            for (int h = 0; h < 32; h++) {
                for (int c = 0; c < 3; c++) {
                    localInput[c / 4][(h * 32 + w) * 4 + c % 4] = input[c][h][w];
                }
            }
        }
        mNnNetwork.predict(localInput);
    }
}
