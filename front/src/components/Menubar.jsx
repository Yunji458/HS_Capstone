function Element({children}){
    return(
      <div>
          <a>{children}</a>
      </div>
    );
}
function Menubar() {
    return(
        <div style={{height: "40px"}}>
            <Element>1content</Element>
            <Element>2content</Element>
            <Element>3content</Element>
        </div>

    );
}

export default Menubar;