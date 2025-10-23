export const Container = {
    display: "flex",
    flexDirection: "row",

};

export const ViewContainer ={
    width: "500px",
    height: "500px",
    backgroundColor: "#c3c3c3",
    borderRadius: "20px",
    margin: "30px",
    flexDirection: "col",
};

export const Button = {
    width: "270px",
    height: "60px",
    backgroundColor: "#5043fd",
    borderRadius: "30px",
    border: 0,
    color: "#ffffff",
    display: "block",
    margin: "0 auto",
    marginBottom: "5px",
};

export const ButtonContainer={
    display: 'block',
    gap: '10px',
    width: '270px',
    margin: '0 auto',
};
export const Video = {
    width: "500px",
};

const commonInput = {
    width: '90%',
    fontSize: "18px",
    backgroundColor: "#f0f8ff",
    borderRadius: "8px",
    border: "1px solid #ccc",

    display: 'block',
    margin: '0 auto',
    marginTop: "50px",
    textAlign: 'center',
}
export const Input ={
    ...commonInput,
    height: "100px",
};

export const InputSentence = {
    ...commonInput,
    height: "150px",
    marginBottom: "70px",
};

