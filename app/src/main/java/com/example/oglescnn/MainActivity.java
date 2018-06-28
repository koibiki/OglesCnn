package com.example.oglescnn;

import android.os.Bundle;
import android.support.v7.app.AppCompatActivity;
import android.view.View;

import com.example.cnnlib.CnnNetwork;
import com.example.cnnlib.layer.ConvolutionLayer;
import com.example.cnnlib.layer.FlatLayer;
import com.example.cnnlib.layer.FullConnectLayer;
import com.example.cnnlib.layer.InputLayer;
import com.example.cnnlib.layer.Layer;
import com.example.cnnlib.layer.NonLinearLayer;
import com.example.cnnlib.layer.PoolingLayer;
import com.example.cnnlib.layer.SoftMaxLayer;

public class MainActivity extends AppCompatActivity {

    private CnnNetwork mCnnNetwork;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

//        buildLetNet();
        buildNet();
//        buildCifaNet();
    }

    @Override
    protected void onStart() {
        super.onStart();
    }

    private void buildLetNet() {
        mCnnNetwork = new CnnNetwork();

        Layer in = new InputLayer(this, new int[]{32, 32, 1});
        mCnnNetwork.addLayer(in);

        Layer conv1 = new ConvolutionLayer(this, in, new int[]{28, 28, 6}, new int[]{5, 5, 1}, 0, new int[]{1, 1}, NonLinearLayer.NonLinearType.RELU);
        mCnnNetwork.addLayer(conv1);

        Layer pool1 = new PoolingLayer(this, conv1, new int[]{14, 14, 6}, new int[]{2, 2}, new int[]{2, 2});
        mCnnNetwork.addLayer(pool1);

        Layer conv2 = new ConvolutionLayer(this, pool1, new int[]{10, 10, 16}, new int[]{5, 5, 6}, 0, new int[]{1, 1}, NonLinearLayer.NonLinearType.RELU);
        mCnnNetwork.addLayer(conv2);

        Layer pool2 = new PoolingLayer(this, conv2, new int[]{5, 5, 16}, new int[]{2, 2}, new int[]{2, 2});
        mCnnNetwork.addLayer(pool2);
//
//        Layer flatLayer = new FlatLayer(this, pool2);
//        mCnnNetwork.addLayer(flatLayer);
//
//        Layer full1 = new FullConnectLayer(this, flatLayer, new int[]{120, 1, 1}, NonLinearLayer.NonLinearType.RELU);
//        mCnnNetwork.addLayer(full1);
//
//        Layer full2 = new FullConnectLayer(this, full1, new int[]{10, 1, 1}, NonLinearLayer.NonLinearType.RELU);
//        mCnnNetwork.addLayer(full2);
//
//        Layer softmaxLayer = new SoftMaxLayer(this, full2, new int[]{10, 1, 1});
//        mCnnNetwork.addLayer(softmaxLayer);

        mCnnNetwork.initialize();
        mCnnNetwork.run();
    }

    private void buildNet() {
        mCnnNetwork = new CnnNetwork();

        Layer in = new InputLayer(this, new int[]{32, 32, 4});
        mCnnNetwork.addLayer(in);

//        Layer test = new TestLayer(this, in, new int[]{32, 32, 4});
//        mCnnNetwork.addLayer(test);

        Layer conv1 = new ConvolutionLayer(this, in, new int[]{32, 32, 64}, new int[]{3, 3, 4}, 1, new int[]{1, 1}, NonLinearLayer.NonLinearType.RELU);
        mCnnNetwork.addLayer(conv1);

        Layer pool1 = new PoolingLayer(this, conv1, new int[]{16, 16, 64}, new int[]{2, 2}, new int[]{2, 2});
        mCnnNetwork.addLayer(pool1);

        Layer conv2 = new ConvolutionLayer(this, pool1, new int[]{16, 16, 256}, new int[]{3, 3, 64}, 1, new int[]{1, 1}, NonLinearLayer.NonLinearType.RELU);
        mCnnNetwork.addLayer(conv2);

        Layer pool2 = new PoolingLayer(this, conv2, new int[]{8, 8, 256}, new int[]{2, 2}, new int[]{2, 2});
        mCnnNetwork.addLayer(pool2);

        Layer conv3 = new ConvolutionLayer(this, pool2, new int[]{8, 8, 256}, new int[]{3, 3, 256}, 1, new int[]{1, 1}, NonLinearLayer.NonLinearType.RELU);
        mCnnNetwork.addLayer(conv3);

        Layer pool3 = new PoolingLayer(this, conv3, new int[]{4, 4, 256}, new int[]{2, 2}, new int[]{2, 2});
        mCnnNetwork.addLayer(pool3);

        Layer flatLayer = new FlatLayer(this, pool3);
        mCnnNetwork.addLayer(flatLayer);

        Layer full1 = new FullConnectLayer(this, flatLayer, 512, NonLinearLayer.NonLinearType.RELU);
        mCnnNetwork.addLayer(full1);

        Layer full2 = new FullConnectLayer(this, full1, 256, NonLinearLayer.NonLinearType.RELU);
        mCnnNetwork.addLayer(full2);

        Layer full3 = new FullConnectLayer(this, full2, 10, NonLinearLayer.NonLinearType.RELU);
        mCnnNetwork.addLayer(full3);

        Layer softmaxLayer = new SoftMaxLayer(this, full2, 10);
        mCnnNetwork.addLayer(softmaxLayer);

        mCnnNetwork.initialize();

        mCnnNetwork.run();

    }

    private void buildCifaNet() {
        mCnnNetwork = new CnnNetwork();

        Layer in = new InputLayer(this, new int[]{32, 32, 3});
        mCnnNetwork.addLayer(in);

//        Layer test = new TestLayer(this, in, new int[]{32, 32, 4});
//        mCnnNetwork.addLayer(test);

        Layer conv1 = new ConvolutionLayer(this, in, new int[]{32, 32, 32}, new int[]{5, 5, 3}, 2, new int[]{1, 1}, NonLinearLayer.NonLinearType.RELU);
        mCnnNetwork.addLayer(conv1);

        Layer pool1 = new PoolingLayer(this, conv1, new int[]{16, 16, 32}, new int[]{2, 2}, new int[]{2, 2});
        mCnnNetwork.addLayer(pool1);

        Layer conv2 = new ConvolutionLayer(this, pool1, new int[]{16, 16, 32}, new int[]{5, 5, 32}, 2, new int[]{1, 1}, NonLinearLayer.NonLinearType.RELU);
        mCnnNetwork.addLayer(conv2);

        Layer pool2 = new PoolingLayer(this, conv2, new int[]{8, 8, 32}, new int[]{2, 2}, new int[]{2, 2});
        mCnnNetwork.addLayer(pool2);

        Layer conv3 = new ConvolutionLayer(this, pool2, new int[]{8, 8, 64}, new int[]{5, 5, 32}, 2, new int[]{1, 1}, NonLinearLayer.NonLinearType.RELU);
        mCnnNetwork.addLayer(conv2);

        Layer pool3 = new PoolingLayer(this, conv2, new int[]{4, 4, 64}, new int[]{2, 2}, new int[]{2, 2});
        mCnnNetwork.addLayer(pool3);

        Layer flatLayer = new FlatLayer(this, pool3);
        mCnnNetwork.addLayer(flatLayer);

        Layer full1 = new FullConnectLayer(this, flatLayer, 64, NonLinearLayer.NonLinearType.RELU);
        mCnnNetwork.addLayer(full1);

        Layer full2 = new FullConnectLayer(this, full1, 10, NonLinearLayer.NonLinearType.RELU);
        mCnnNetwork.addLayer(full2);

        Layer full3 = new FullConnectLayer(this, full2, 10, NonLinearLayer.NonLinearType.RELU);
        mCnnNetwork.addLayer(full3);


//        Layer softmaxLayer = new SoftMaxLayer(this, full2, new int[]{20, 1, 1});
//        mCnnNetwork.addLayer(softmaxLayer);

        mCnnNetwork.initialize();
            mCnnNetwork.run();

    }

    public void performCNN(View view) {
        mCnnNetwork.run();
    }

    public void showOutput(View view) {
        mCnnNetwork.readOutput();
    }
}
