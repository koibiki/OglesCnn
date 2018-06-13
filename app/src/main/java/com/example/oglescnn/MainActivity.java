package com.example.oglescnn;

import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.view.View;

import com.example.cnnlib.CnnNetwork;
import com.example.cnnlib.activate.ActivationFunc;
import com.example.cnnlib.layer.ConvolutionLayer;
import com.example.cnnlib.layer.InputLayer;
import com.example.cnnlib.layer.Layer;
import com.example.cnnlib.model.Kennel;
import com.example.cnnlib.model.LayerParams;
import com.example.cnnlib.utils.Constants;
import com.example.cnnlib.utils.MathUtils;

import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.List;

public class MainActivity extends AppCompatActivity {

    private CnnNetwork mCnnNetwork;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
    }

    @Override
    protected void onStart() {
        super.onStart();
        mCnnNetwork = new CnnNetwork();
        LayerParams inLayerParams = new LayerParams(32, 32, 4, 32, 32, 4);
        mCnnNetwork.addLayer(new InputLayer(this, inLayerParams));


        LayerParams convParams1 = new LayerParams(32, 32, 4, 32, 32, 2);
        List<Kennel> kennels = new ArrayList<>();
        for (int i = 0; i < 2; i++) {
            kennels.add(createKennel(i + 1, 5, 4));
        }
        mCnnNetwork.addLayer(new ConvolutionLayer(this, convParams1, kennels, 1, new int[]{1, 1}));


//        LayerParams convParams2 = new LayerParams(32, 32, 4, 32, 32, 200);
//        List<Kennel> kennels2 = new ArrayList<>();
//        for (int i = 0; i < 200; i++) {
//            kennels2.add(createKennel(i + 1, 3, 64));
//        }
//        mCnnNetwork.addLayer(new ConvolutionLayer(this, convParams2, kennels2, 1, Constants.S_PADDING_SAME, new int[]{1, 1}));

    }

    private Kennel createKennel(int num, int length, int channel) {
        float[] data = new float[length * length * channel];
        for (int i = 0; i < length * length * channel; i++) {
            data[i] = num;
        }
        return new Kennel(length, length, channel, data);
    }

    public void performCNN(View view) {
        mCnnNetwork.run();
    }
}
