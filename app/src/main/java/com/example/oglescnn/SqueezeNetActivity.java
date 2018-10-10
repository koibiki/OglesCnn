package com.example.oglescnn;

import android.os.Bundle;
import android.os.Handler;
import android.os.HandlerThread;
import android.support.v7.app.AppCompatActivity;
import android.view.View;
import android.widget.ImageView;
import android.widget.ProgressBar;
import android.widget.TextView;

import com.example.eglnn.NnNetwork;
import com.example.eglnn.layer.Concat;
import com.example.eglnn.layer.ConvGEMM;
import com.example.eglnn.layer.Flat;
import com.example.eglnn.layer.GlobalAvePooling;
import com.example.eglnn.layer.Input;
import com.example.eglnn.layer.Layer;
import com.example.eglnn.layer.Layer.PaddingType;
import com.example.eglnn.layer.Pooling;
import com.example.eglnn.layer.SoftMax;
import com.example.eglnn.layer.SoftMaxCpu;
import com.example.eglnn.utils.Numpy;
import com.example.eglnn.utils.Utils;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;

public class SqueezeNetActivity extends AppCompatActivity {

    private NnNetwork mNnNetwork;
    private String[] mLabels;
    private ImageView iv;
    private TextView tv;
    private TextView tvPb;
    private View pb;
    private Handler mHandler;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        iv = findViewById(R.id.iv);
        tv = findViewById(R.id.tv);
        pb = findViewById(R.id.pb);
        tvPb = findViewById(R.id.tv_pb);
        iv.setImageResource(R.drawable.cat217);
        mLabels = readLabel();

        HandlerThread mGLThread = new HandlerThread("GL");
        mGLThread.start();
        mHandler = new Handler(mGLThread.getLooper());
        mHandler.post(this::buildSqueezeNet2);
    }

    private String[] readLabel() {
        String[] splits = null;
        try {
            InputStream open = getAssets().open("squeezenet/labels.txt");
            String labels = readTextFromIn(open);
            splits = labels.split("\n");
        } catch (IOException e) {
            e.printStackTrace();
        } catch (Exception e) {
            e.printStackTrace();
        }
        return splits;
    }

    private String readTextFromIn(InputStream is) throws Exception {
        InputStreamReader reader = new InputStreamReader(is);
        BufferedReader bufferedReader = new BufferedReader(reader);
        StringBuffer buffer = new StringBuffer("");
        String str;
        while ((str = bufferedReader.readLine()) != null) {
            buffer.append(str.trim());
            buffer.append("\n");
        }
        return buffer.toString();
    }

    private void buildSqueezeNet2() {
        mNnNetwork = new NnNetwork(this);

        String asssetDir = "squeezenet/";

        Layer in = new Input(this, "input", 227, 227, 3);
        mNnNetwork.addLayer(in);

        Layer conv1 = new ConvGEMM(this, asssetDir + "conv1", in, 64, 3, 3, PaddingType.VALID, 2, 2, Layer.ActiveType.RELU);
        mNnNetwork.addLayer(conv1);

        Layer pooling1 = new Pooling(this, "pool1", conv1, 3, 3, PaddingType.VALID, 2, 2, Layer.PoolingType.MAX);
        mNnNetwork.addLayer(pooling1);

        // fire2
        Layer conv2_squeeze = new ConvGEMM(this, asssetDir + "fire2_squeeze1x1", pooling1, 16, 1, 1, PaddingType.SAME, 1, 1, Layer.ActiveType.RELU);
        mNnNetwork.addLayer(conv2_squeeze);

        Layer conv2_1 = new ConvGEMM(this, asssetDir + "fire2_expand1x1", conv2_squeeze, 64, 1, 1, PaddingType.SAME, 1, 1, Layer.ActiveType.RELU);
        mNnNetwork.addLayer(conv2_1);

        Layer conv2_2 = new ConvGEMM(this, asssetDir + "fire2_expand3x3", conv2_squeeze, 64, 3, 3, PaddingType.SAME, 1, 1, Layer.ActiveType.RELU);
        mNnNetwork.addLayer(conv2_2);

        Layer concat2 = new Concat(this, "concat2", new Layer[]{conv2_1, conv2_2}, 2);
        mNnNetwork.addLayer(concat2);

        // fire3
        Layer conv3_squeeze = new ConvGEMM(this, asssetDir + "fire3_squeeze1x1", concat2, 16, 1, 1, PaddingType.SAME, 1, 1, Layer.ActiveType.RELU);
        mNnNetwork.addLayer(conv3_squeeze);

        Layer conv3_1 = new ConvGEMM(this, asssetDir + "fire3_expand1x1", conv3_squeeze, 64, 1, 1, PaddingType.SAME, 1, 1, Layer.ActiveType.RELU);
        mNnNetwork.addLayer(conv3_1);

        Layer conv3_2 = new ConvGEMM(this, asssetDir + "fire3_expand3x3", conv3_squeeze, 64, 3, 3, PaddingType.SAME, 1, 1, Layer.ActiveType.RELU);
        mNnNetwork.addLayer(conv3_2);

        Layer concat3 = new Concat(this, "concat3", new Layer[]{conv3_1, conv3_2}, 2);
        mNnNetwork.addLayer(concat3);

        Layer pooling3 = new Pooling(this, "pool3", concat3, 3, 3, PaddingType.VALID, 2, 2, Layer.PoolingType.MAX);
        mNnNetwork.addLayer(pooling3);

        // fire4
        Layer conv4_squeeze = new ConvGEMM(this, asssetDir + "fire4_squeeze1x1", pooling3, 32, 1, 1, PaddingType.SAME, 1, 1, Layer.ActiveType.RELU);
        mNnNetwork.addLayer(conv4_squeeze);

        Layer conv4_1 = new ConvGEMM(this, asssetDir + "fire4_expand1x1", conv4_squeeze, 128, 1, 1, PaddingType.SAME, 1, 1, Layer.ActiveType.RELU);
        mNnNetwork.addLayer(conv4_1);

        Layer conv4_2 = new ConvGEMM(this, asssetDir + "fire4_expand3x3", conv4_squeeze, 128, 3, 3, PaddingType.SAME, 1, 1, Layer.ActiveType.RELU);
        mNnNetwork.addLayer(conv4_2);

        Concat concat4 = new Concat(this, "concat4", new Layer[]{conv4_1, conv4_2}, 2);
        mNnNetwork.addLayer(concat4);

        // fire5
        Layer conv5_squeeze = new ConvGEMM(this, asssetDir + "fire5_squeeze1x1", concat4, 32, 1, 1, PaddingType.SAME, 1, 1, Layer.ActiveType.RELU);
        mNnNetwork.addLayer(conv5_squeeze);

        Layer conv5_1 = new ConvGEMM(this, asssetDir + "fire5_expand1x1", conv5_squeeze, 128, 1, 1, PaddingType.SAME, 1, 1, Layer.ActiveType.RELU);
        mNnNetwork.addLayer(conv5_1);

        Layer conv5_2 = new ConvGEMM(this, asssetDir + "fire5_expand3x3", conv5_squeeze, 128, 3, 3, PaddingType.SAME, 1, 1, Layer.ActiveType.RELU);
        mNnNetwork.addLayer(conv5_2);

        Concat concat5 = new Concat(this, "concat5", new Layer[]{conv5_1, conv5_2}, 2);
        mNnNetwork.addLayer(concat5);

        Layer pooling5 = new Pooling(this, "pool5", concat5, 3, 3, PaddingType.VALID, 2, 2, Layer.PoolingType.MAX);
        mNnNetwork.addLayer(pooling5);

        // fire6
        Layer conv6_squeeze = new ConvGEMM(this, asssetDir + "fire6_squeeze1x1", pooling5, 48, 1, 1, PaddingType.SAME, 1, 1, Layer.ActiveType.RELU);
        mNnNetwork.addLayer(conv6_squeeze);

        Layer conv6_1 = new ConvGEMM(this, asssetDir + "fire6_expand1x1", conv6_squeeze, 192, 1, 1, PaddingType.SAME, 1, 1, Layer.ActiveType.RELU);
        mNnNetwork.addLayer(conv6_1);

        Layer conv6_2 = new ConvGEMM(this, asssetDir + "fire6_expand3x3", conv6_squeeze, 192, 3, 3, PaddingType.SAME, 1, 1, Layer.ActiveType.RELU);
        mNnNetwork.addLayer(conv6_2);

        Layer concat6 = new Concat(this, "concat6", new Layer[]{conv6_1, conv6_2}, 2);
        mNnNetwork.addLayer(concat6);

        // fire7
        Layer conv7_squeeze = new ConvGEMM(this, asssetDir + "fire7_squeeze1x1", concat6, 48, 1, 1, PaddingType.SAME, 1, 1, Layer.ActiveType.RELU);
        mNnNetwork.addLayer(conv7_squeeze);

        Layer conv7_1 = new ConvGEMM(this, asssetDir + "fire7_expand1x1", conv7_squeeze, 192, 1, 1, PaddingType.SAME, 1, 1, Layer.ActiveType.RELU);
        mNnNetwork.addLayer(conv7_1);

        Layer conv7_2 = new ConvGEMM(this, asssetDir + "fire7_expand3x3", conv7_squeeze, 192, 3, 3, PaddingType.SAME, 1, 1, Layer.ActiveType.RELU);
        mNnNetwork.addLayer(conv7_2);

        Layer concat7 = new Concat(this, "concat7", new Layer[]{conv7_1, conv7_2}, 2);
        mNnNetwork.addLayer(concat7);

        // fire8
        Layer conv8_squeeze = new ConvGEMM(this, asssetDir + "fire8_squeeze1x1", concat7, 64, 1, 1, PaddingType.SAME, 1, 1, Layer.ActiveType.RELU);
        mNnNetwork.addLayer(conv8_squeeze);

        Layer conv8_1 = new ConvGEMM(this, asssetDir + "fire8_expand1x1", conv8_squeeze, 256, 1, 1, PaddingType.SAME, 1, 1, Layer.ActiveType.RELU);
        mNnNetwork.addLayer(conv8_1);

        Layer conv8_2 = new ConvGEMM(this, asssetDir + "fire8_expand3x3", conv8_squeeze, 256, 3, 3, PaddingType.SAME, 1, 1, Layer.ActiveType.RELU);
        mNnNetwork.addLayer(conv8_2);

        Layer concat8 = new Concat(this, "concat8", new Layer[]{conv8_1, conv8_2}, 2);
        mNnNetwork.addLayer(concat8);

        // fire9
        Layer conv9_squeeze = new ConvGEMM(this, asssetDir + "fire9_squeeze1x1", concat8, 64, 1, 1, PaddingType.SAME, 1, 1, Layer.ActiveType.RELU);
        mNnNetwork.addLayer(conv9_squeeze);

        Layer conv9_1 = new ConvGEMM(this, asssetDir + "fire9_expand1x1", conv9_squeeze, 256, 1, 1, PaddingType.SAME, 1, 1, Layer.ActiveType.RELU);
        mNnNetwork.addLayer(conv9_1);

        Layer conv9_2 = new ConvGEMM(this, asssetDir + "fire9_expand3x3", conv9_squeeze, 256, 3, 3, PaddingType.SAME, 1, 1, Layer.ActiveType.RELU);
        mNnNetwork.addLayer(conv9_2);

        Layer concat9 = new Concat(this, "concat9", new Layer[]{conv9_1, conv9_2}, 2);
        mNnNetwork.addLayer(concat9);

        // fire10
        Layer conv10 = new ConvGEMM(this, asssetDir + "conv10", concat9, 1000, 1, 1, PaddingType.SAME, 1, 1, Layer.ActiveType.NONE);
        mNnNetwork.addLayer(conv10);

        Layer pooling11 = new GlobalAvePooling(this, "pool11", conv10);
        mNnNetwork.addLayer(pooling11);

        Layer flat = new Flat(this, "flat", pooling11);
        mNnNetwork.addLayer(flat);

        Layer softmax = new SoftMax(this, "softmax", flat);
        mNnNetwork.addLayer(softmax);

        mNnNetwork.initialize();

        pb.setVisibility(View.INVISIBLE);

    }

    public void runNn(View view) {
        pb.setVisibility(View.VISIBLE);
        tvPb.setText("识别中");
        tv.setText("识别中");
        mHandler.post(() -> {
            float[][][] input = TestUtils.getTestImage(this, 227, 227);
            float[][] localInput = new float[Utils.alignBy4(3) / 4][227 * 227 * 4];
            for (int w = 0; w < 227; w++) {
                for (int h = 0; h < 227; h++) {
                    for (int c = 0; c < 3; c++) {
                        localInput[c / 4][(h * 227 + w) * 4 + c % 4] = input[c][h][w];
                    }
                }
            }
            NnNetwork.Result predict = mNnNetwork.predict(localInput);
            float[] result = Numpy.argmax(predict.getResult());
            String label = mLabels[(int) result[0]];
            runOnUiThread(() -> {
                pb.setVisibility(View.INVISIBLE);
                tv.setText("label:" + label + "\n" + "accuracy :" + result[1] + "\n" + "avetime:" + predict.getAveTime());
            });
        });

    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        mNnNetwork.destory();
    }
}
