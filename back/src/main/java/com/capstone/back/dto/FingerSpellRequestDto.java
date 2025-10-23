package com.capstone.back.dto;

import java.util.List;

public class FingerSpellRequestDto {
    private String action;
    private List<List<Float>> sequence;

    public String getAction() {
        return action;
    }

    public void setAction(String action) {
        this.action = action;
    }

    public List<List<Float>> getSequence() {
        return sequence;
    }

    public void setSequence(List<List<Float>> sequence) {
        this.sequence = sequence;
    }
}
