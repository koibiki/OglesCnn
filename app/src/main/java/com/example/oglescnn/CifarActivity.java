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
import com.example.eglnn.layer.ConvGEMM;
import com.example.eglnn.layer.Flat;
import com.example.eglnn.layer.FullConnSSBO;
import com.example.eglnn.layer.Input;
import com.example.eglnn.layer.Layer;
import com.example.eglnn.layer.Layer.PaddingType;
import com.example.eglnn.layer.Pooling;
import com.example.eglnn.layer.SoftMax;
import com.example.eglnn.utils.Numpy;
import com.example.eglnn.utils.Utils;

import java.io.BufferedInputStream;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;

public class CifarActivity extends AppCompatActivity {

    private NnNetwork mNnNetwork;
    private String[] mLabels;
    private ImageView iv;
    private TextView tv;
    private View pb;
    private TextView tvPb;
    private Handler mHandler;
    private View run;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        iv = findViewById(R.id.iv);
        tv = findViewById(R.id.tv);
        pb = findViewById(R.id.pb);
        tvPb = findViewById(R.id.tv_pb);
        run = findViewById(R.id.run);
        iv.setImageResource(R.drawable.airplane);
        mLabels = readLabel();

        HandlerThread mGLThread = new HandlerThread("GL");
        mGLThread.start();
        mHandler = new Handler(mGLThread.getLooper());
        mHandler.post(this::buildCifar10Net);
    }

    private String[] readLabel() {
        String[] splits = null;
        try {
            InputStream open = getAssets().open("cifar10/labels.txt");
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


    private void buildCifar10Net() {
        mNnNetwork = new NnNetwork(this);

        String asssetDir = "cifar10/";

        Layer in = new Input(this, "input", 32, 32, 3);
        mNnNetwork.addLayer(in);

        Layer conv1 = new ConvGEMM(this, asssetDir + "conv1", in, 32, 5, 5, PaddingType.SAME, 1, 1, Layer.ActiveType.RELU);
        mNnNetwork.addLayer(conv1);

        Pooling pooling1 = new Pooling(this, "pool1", conv1, 3, 3, PaddingType.SAME, 2, 2, Layer.PoolingType.MAX);
        mNnNetwork.addLayer(pooling1);

        Layer conv2 = new ConvGEMM(this, asssetDir + "conv2", pooling1, 32, 5, 5, PaddingType.SAME, 1, 1, Layer.ActiveType.RELU);
        mNnNetwork.addLayer(conv2);

        Pooling pooling2 = new Pooling(this, "pool2", conv2, 3, 3, PaddingType.SAME, 2, 2, Layer.PoolingType.MAX);
        mNnNetwork.addLayer(pooling2);

        Layer conv3 = new ConvGEMM(this, asssetDir + "conv3", pooling2, 64, 5, 5, PaddingType.SAME, 1, 1, Layer.ActiveType.RELU);
        mNnNetwork.addLayer(conv3);

        Pooling pooling3 = new Pooling(this, "pool2", conv3, 3, 3, PaddingType.SAME, 2, 2, Layer.PoolingType.MAX);
        mNnNetwork.addLayer(pooling3);

        Layer fc1 = new FullConnSSBO(this, asssetDir + "fc1", pooling3, 64, Layer.ActiveType.NONE);
        mNnNetwork.addLayer(fc1);

        Layer fc2 = new FullConnSSBO(this, asssetDir + "fc2", fc1, 10, Layer.ActiveType.NONE);
        mNnNetwork.addLayer(fc2);

        Flat flat = new Flat(this, "flat", fc2);
        mNnNetwork.addLayer(flat);

        SoftMax softmax = new SoftMax(this, "softmax", flat);
        mNnNetwork.addLayer(softmax);

        mNnNetwork.initialize();

        runOnUiThread(()->{
            pb.setVisibility(View.INVISIBLE);
            run.setEnabled(true);
        });
    }

    public void runNn(View view) {
        run.setEnabled(false);

        pb.setVisibility(View.VISIBLE);
        tvPb.setText("识别中");
        tv.setText("识别中");
        mHandler.post(() -> {
            float[][][] input = TestUtils.getTestCifarImage(this);
            float[][] localInput = new float[Utils.alignBy4(3) / 4][32 * 32 * 4];
            for (int w = 0; w < 32; w++) {
                for (int h = 0; h < 32; h++) {
                    for (int c = 0; c < 3; c++) {
                        localInput[c / 4][(h * 32 + w) * 4 + c % 4] = input[c][h][w];
                    }
                }
            }
            NnNetwork.Result predict = mNnNetwork.predict(localInput);
            float[] result = Numpy.argmax(predict.getResult());
            String label = mLabels[(int) result[0]];
            runOnUiThread(() -> {
                run.setEnabled(true);
                pb.setVisibility(View.GONE);
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
