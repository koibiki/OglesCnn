package com.example.oglescnn;

import android.os.Bundle;
import android.os.Environment;
import android.support.v7.app.AppCompatActivity;
import android.view.View;

import com.example.cnnlib.CnnNetwork;
import com.example.cnnlib.layer.ConvSSBO;
import com.example.cnnlib.layer.Flat;
import com.example.cnnlib.layer.FullConnSSBO;
import com.example.cnnlib.layer.Input;
import com.example.cnnlib.layer.Layer;
import com.example.cnnlib.layer.NonLinear;
import com.example.cnnlib.layer.Pooling;
import com.example.cnnlib.layer.SoftMax;
import com.example.cnnlib.utils.DataUtils;

public class MainActivity extends AppCompatActivity {

    private CnnNetwork mCnnNetwork;
    private String sRoot = Environment.getExternalStorageDirectory().getAbsolutePath();
    private String TAG = "MainActivity";


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

//        buildNet();
//        buildLetNet();
        buildTestNet();
    }

    private void buildTestNet() {
        mCnnNetwork = new CnnNetwork(this);

        Layer in = new Input(this, 32, 32, 3);
        mCnnNetwork.addLayer(in);

        Layer conv1 = new ConvSSBO(this, in, 6, 5, 5, 2, 1, 1, NonLinear.NonLinearType.RELU, "");
        mCnnNetwork.addLayer(conv1);

        Layer pool1 = new Pooling(this, conv1, 2, 2, 2, 2);
        mCnnNetwork.addLayer(pool1);

        Layer conv2 = new ConvSSBO(this, pool1, 16, 5, 5, 2, 1, 1, NonLinear.NonLinearType.RELU, "");
        mCnnNetwork.addLayer(conv2);

        mCnnNetwork.initialize();
    }


    private void buildLetNet() {
        mCnnNetwork = new CnnNetwork(this);

        Layer in = new Input(this, 32, 32, 1);
        mCnnNetwork.addLayer(in);

        Layer conv1 = new ConvSSBO(this, in, 6, 5, 5, 0, 1, 1, NonLinear.NonLinearType.RELU, "");
        mCnnNetwork.addLayer(conv1);

        Layer pool1 = new Pooling(this, conv1, 2, 2, 2, 2);
        mCnnNetwork.addLayer(pool1);

        Layer conv2 = new ConvSSBO(this, pool1, 16, 5, 5, 0, 1, 1, NonLinear.NonLinearType.RELU, "");
        mCnnNetwork.addLayer(conv2);

        Layer pool2 = new Pooling(this, conv2, 2, 2, 2, 2);
        mCnnNetwork.addLayer(pool2);

        Layer flat = new Flat(this, pool2);
        mCnnNetwork.addLayer(flat);

        Layer full1 = new FullConnSSBO(this, flat, 120, NonLinear.NonLinearType.RELU, "");
        mCnnNetwork.addLayer(full1);

        Layer full2 = new FullConnSSBO(this, full1, 10, NonLinear.NonLinearType.RELU, "");
        mCnnNetwork.addLayer(full2);

        Layer softmax = new SoftMax(this, full2, 10);
        mCnnNetwork.addLayer(softmax);

        mCnnNetwork.initialize();
    }

    private void buildNet() {
        try {
            mCnnNetwork = new CnnNetwork(this, sRoot + "/Cifar10es/net/Cifar10_def_test.txt");
            mCnnNetwork.initialize();
        } catch (Exception e) {
            e.printStackTrace();
        }


    }

    public void performCNN(View view) {
        float[][][] input = DataUtils.createInputBuffer(new int[]{32, 32, 3});
//        float[][][] input = testUtils.getTestImage(this);
        mCnnNetwork.predict(input);
    }

    public void showOutput(View view) {
        mCnnNetwork.readOutput();
    }
}
