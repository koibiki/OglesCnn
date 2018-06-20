package com.example.cnnlib.layer;

import android.content.Context;

import com.example.cnnlib.model.LayerParams;
import com.example.cnnlib.render.ComputeRender;
import com.example.cnnlib.utils.AttachIDManager;
import com.example.cnnlib.utils.DataUils;

import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.List;

import static com.example.cnnlib.render.ComputeRender.getCompShaderLocalSizeY;
import static com.example.cnnlib.render.ComputeRender.initPoolingPro;

/**
 * kennel 存储方式为 纹理每行存储一个 kennel， 纹理高度与kennel个数相同
 */
public class FullConnectLayer extends Layer {

    private static final String TAG = "FullConnectLayer";

    List<float[]> mKennels = new ArrayList<>();
    private int mNumGroupsY;
    private int mShaderPro;
    private int mKennelAmount;                      // kennel组 个数

    private int[] mParams;
    private int mKennelTex;

    public FullConnectLayer(Context context, LayerParams layerParams, int kennelAmount) {
        super(context, layerParams);
        this.mKennelAmount = kennelAmount;

        initFullConnect();
    }

    private void initFullConnect() {
        int localSizeY = getCompShaderLocalSizeY(mLayerParams.outputShape);
        mNumGroupsY = (int) Math.ceil(mLayerParams.outputShape[1] * 1.0d / localSizeY);
        mShaderPro = initPoolingPro(mContext, "full_connect.comp", mKennelAmount, mLayerParams.outputShape[0], localSizeY);
        mAttachID = AttachIDManager.getInstance().getAttachID();
        mOutTex = ComputeRender.createTexture(mAttachID);


        int[] kennelSize = calculateKennelSize();
        int attachID = AttachIDManager.getInstance().getAttachID();
        mKennelTex = ComputeRender.createTexture(attachID, kennelSize[0], mKennelAmount);
        createKennels();
        transferKennelToTex(kennelSize, mKennelTex);

        mParams = new int[9];
        mParams[0] = mLayerParams.inputShape[0];
        mParams[1] = mLayerParams.inputShape[1];
        mParams[2] = mLayerParams.inputShape[2];
        mParams[3] = mLayerParams.outputShape[0];
        mParams[4] = mLayerParams.outputShape[1];
        mParams[5] = mLayerParams.outputShape[2];
        mParams[6] = kennelSize[0];
        mParams[7] = kennelSize[1];
        mParams[8] = kennelSize[2];
    }

    // kennel size width <= 1024 height=1
    private int[] calculateKennelSize() {
        int[] kennelSize = new int[3];
        int[] inputShape = mLayerParams.inputShape;
        int inputSize = inputShape[0] * inputShape[1] * inputShape[2] + 1;  // 最后一列存储bias
        int width = inputSize / 4;
        int remain = inputSize % 4;
        kennelSize[0] = remain == 0 ? width : width + 1;
        kennelSize[1] = 1;
        kennelSize[2] = remain - 1 < 0 ? 3 : remain - 1;    // 表示最后一位数据的下标
        return kennelSize;
    }

    private void transferKennelToTex(int[] kennelSize, int tex) {
        for (int i = 0; i < mKennels.size(); i++) {
            ComputeRender.transferToTexture(FloatBuffer.wrap(mKennels.get(i)), tex, 0, i, kennelSize[0], 1);
        }
    }

    private void createKennels() {
        int[] inputShape = mLayerParams.inputShape;
        int inputSize = inputShape[0] * inputShape[1] * inputShape[2] + 1;     // 最后一列存储bias
        for (int i = 0; i < mKennelAmount; i++) {
            float[] kennel = DataUils.createFullConnKennel(inputSize, i);
            mKennels.add(kennel);
        }
    }

    @Override
    public int forwardProc(int inTex) {
        ComputeRender.performFullConnect(mShaderPro, mParams, inTex, mOutTex, mKennelTex, mNumGroupsY);
        return mOutTex;
    }

    @Override
    public void readOutput() {

    }

}
