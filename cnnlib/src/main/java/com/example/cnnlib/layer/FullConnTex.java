package com.example.cnnlib.layer;

import android.content.Context;
import android.text.TextUtils;

import com.example.cnnlib.render.Render;
import com.example.cnnlib.utils.DataUtils;
import com.example.cnnlib.utils.NetUtils;
import com.example.cnnlib.utils.ParamUnpacker;

import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.List;

import static com.example.cnnlib.render.Render.initFullConnPro;

/**
 * 全连接前接 flat 或 全连接层 kennel 个数不超过4096
 * 全连接层的 输入 输出 的channel 和 kennel 都需要 4 对齐
 * kennel 存储次序为 按截面坐标顺序 依次填充channel
 */
public class FullConnTex extends Layer {

    private static final String TAG = "FullConnSSBO";

    private List<float[]> mKennels;
    private int mPro;
    private int mKennelAmount;
    private String mParamFilePath;

    private int[] mParams;

    private NonLinear.Type mType;

    private int mStartY;
    private int mKennelAttachId;
    private int mKennelTex;
    private int mNumGroupZ;

    public FullConnTex(Context context, Layer preLayer, int kennelAmount, NonLinear.Type type, String paramFilePath) {
        super(context, preLayer);
        this.mKennelAmount = kennelAmount;
        this.mType = type;
        this.mOutputShape = calculateFullShape(kennelAmount);
        this.mParamFilePath = paramFilePath;
    }

    private int[] calculateFullShape(int kennelAmount) {
        return new int[]{1, 1, kennelAmount};
    }

    @Override
    public void initialize() {
        initFullConnect();
    }

    private void initFullConnect() {
        int[] inputShape = mPreLayer.getOutputShape();
        int kennelSize = inputShape[0] * inputShape[1] * inputShape[2] + 1;
        int alignKennelSize =
                inputShape[0] * inputShape[1] * NetUtils.alignBy4(inputShape[2]) + 4;

        int local_size_z;
        if (NetUtils.alignBy4(mKennelAmount) / 4 <= 64) {
            local_size_z = NetUtils.alignBy4(mKennelAmount) / 4;
            mNumGroupZ = 1;
        } else {
            local_size_z = 64;
            mNumGroupZ = (int) Math.ceil(NetUtils.alignBy4(mKennelAmount) / 4 / 64);
        }

        mPro = initFullConnPro(mContext, "full_conn_tex.comp", alignKennelSize, mKennelAmount, 1, 1,  local_size_z);
        mAttachID = Layer.getDataAttachID();
        mOutTex = Render.createTexture();

        mKennelAttachId = Layer.getConvKennelAttachID();
        mKennelTex = Render.getConvKennelTexture();


        mStartY = Layer.getConvStartY(mKennelAmount);
        if (TextUtils.isEmpty(mParamFilePath)) {
            mKennels = createKennels(alignKennelSize);
        } else {
            mKennels = loadKennels(kennelSize, alignKennelSize);
        }
        transferKennelToTex();

        mParams = new int[8];
        mParams[0] = inputShape[0];
        mParams[1] = inputShape[1];
        mParams[2] = inputShape[2];
        mParams[3] = mOutputShape[0];
        mParams[4] = mOutputShape[1];
        mParams[5] = mOutputShape[2];
        mParams[6] = mType.index;
        mParams[7] = mStartY;

    }

    private List<float[]> loadKennels(int kennelSize, int alignKennelSize) {
        ParamUnpacker paramUnpacker = new ParamUnpacker();
        Object[] objects = paramUnpacker.unpackerFunction(mParamFilePath, new Class[]{float[].class, float[].class});
        float[] weight = (float[]) objects[0];
        float[] bias = (float[]) objects[1];

        int wSize = kennelSize - 1;

        List<float[]> kennels = new ArrayList<>();
        for (int i = 0; i < mKennelAmount; i++) {
            float[] kennel = new float[alignKennelSize];
            for (int s = 0; s < wSize; s++) {
                int wIndex = i * wSize + s;
                kennel[s] = weight[wIndex];
            }
            kennel[alignKennelSize - 4] = bias[i];
            kennels.add(kennel);
        }
        return kennels;
    }

    private void transferKennelToTex() {
        Render.bindTextureAndBuffer(mKennelTex, mKennelAttachId);
        for (int i = 0; i < mKennels.size(); i++) {
            float[] kennel = mKennels.get(i);
            Render.transferToTexture(FloatBuffer.wrap(kennel), mKennelTex, 0, mStartY + i, kennel.length / 4, 1);
        }
    }

    private List<float[]> createKennels(int alignSize) {
        List<float[]> kennels = new ArrayList<>();
        for (int i = 0; i < mKennelAmount; i++) {
            float[] kennel = DataUtils.createFullConnKennel(alignSize, mPreLayer.getOutputShape(), i);
            kennels.add(kennel);
        }
        return kennels;
    }

    @Override
    protected void bindTextureAndBuffer() {
        Render.bindTextureAndBuffer(mOutTex, mAttachID);
    }

    @Override
    protected void actualForwardProc(float[][][] input) {
        Render.performFullConnectTex(mPro, mParams, mPreLayer.getOutTex(), mOutTex, mKennelTex, mNumGroupZ);
    }

}
